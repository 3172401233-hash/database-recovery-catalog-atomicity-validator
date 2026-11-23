"""恢复目录原子性验证器。

在数据库崩溃后,根据目录版本、双指针、检查点、WAL 段、基础备份、
摘要清单和更新事务,判断哪个目录版本构成完整的恢复点。

核心原则:目录更新必须与文件持久化边界(fsync)一致,
不能只依赖目录记录中的时间戳判断合法性。

注意:本模块中所有对象存储、WAL、基础备份均为内存模拟,
不访问任何真实存储系统。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# 输入模型(全部为模拟对象)
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    """模拟对象存储中的一个文件及其持久化状态。"""

    file_id: str
    txn_id: str
    persisted: bool  # 是否已完成 fsync(持久化边界)
    digest: Optional[str] = None


@dataclass
class CatalogVersion:
    """一个目录版本:对象名 -> 文件 id 的映射,加上恢复锚点。"""

    version: int
    txn_id: str
    entries: Dict[str, str]  # 对象名 -> file_id
    checkpoint_id: str = ""
    wal_start: int = 0  # 该版本可见数据所需的最早 WAL 位置
    wal_end: int = 0    # 该版本可见数据所需的最新 WAL 位置(不含)


@dataclass
class DualPointer:
    """双指针(主/镜像)。目录切换必须两份都落盘才算完成。"""

    primary: int
    mirror: int
    high_water_mark: int  # 历史上已提交过的最大版本号,用于识别旧指针回退


@dataclass
class Checkpoint:
    checkpoint_id: str
    wal_start: int
    base_backup_id: Optional[str] = None


@dataclass
class WALSegment:
    segment_id: str
    start_lsn: int
    end_lsn: int  # 不含


@dataclass
class BaseBackup:
    backup_id: str
    catalog_version: int  # 该基础备份对应的目录版本
    checkpoint_id: str = ""


@dataclass
class DigestManifest:
    """已持久化文件的摘要清单。缺失摘要 = 文件未确认落盘。"""

    digests: Dict[str, str]  # file_id -> digest


@dataclass
class UpdateTransaction:
    """一次目录更新事务:先持久化文件,再原子切换目录指针。"""

    txn_id: str
    status: str  # 'committed' | 'partial' | 'aborted'
    fsynced_files: List[str] = field(default_factory=list)
    catalog_version: Optional[int] = None


# ---------------------------------------------------------------------------
# 输出模型
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    recoverable_version: Optional[int]
    orphaned_objects: List[str]
    blocking_reasons: List[str]
    version_status: Dict[int, List[str]]

    @property
    def recoverable(self) -> bool:
        return self.recoverable_version is not None


# ---------------------------------------------------------------------------
# 验证器
# ---------------------------------------------------------------------------

def _txn_version_of(txn):
    return txn.catalog_version if txn is not None else None


class CatalogAtomicityValidator:
    """对崩溃后的目录状态做原子性验证,找出完整恢复点。"""

    def __init__(self,
                 catalog_versions,
                 pointer,
                 checkpoints,
                 wal_segments,
                 base_backups,
                 manifest,
                 transactions,
                 files):
        # type: (List[CatalogVersion], DualPointer, List[Checkpoint], List[WALSegment], List[BaseBackup], DigestManifest, List[UpdateTransaction], List[FileRecord]) -> None
        self.versions = {v.version: v for v in catalog_versions}
        self.pointer = pointer
        self.checkpoints = {c.checkpoint_id: c for c in checkpoints}
        self.wal_segments = list(wal_segments)
        self.base_backups = {b.backup_id: b for b in base_backups}
        self.manifest = manifest
        self.transactions = {t.txn_id: t for t in transactions}
        self.files = {f.file_id: f for f in files}

    # -- 单项检查 ---------------------------------------------------------

    def _check_transaction(self, v):
        reasons = []
        txn = self.transactions.get(v.txn_id)
        if txn is None:
            reasons.append("unknown_transaction:%s" % v.txn_id)
            return reasons
        if txn.status == "partial":
            reasons.append("partial_write:事务 %s 未完整提交" % v.txn_id)
        elif txn.status == "aborted":
            reasons.append("aborted_transaction:%s" % v.txn_id)
        elif txn.status != "committed":
            reasons.append("invalid_transaction_status:%s" % txn.status)
        return reasons

    def _check_files(self, v):
        """目录引用的每个文件都必须越过持久化边界且有摘要。"""
        reasons = []
        committed_versions = self._committed_versions()
        for name, file_id in sorted(v.entries.items()):
            f = self.files.get(file_id)
            if f is None:
                reasons.append("missing_file:%s->%s" % (name, file_id))
                continue
            owner_txn = self.transactions.get(f.txn_id)
            owner_fsynced = set(owner_txn.fsynced_files) if owner_txn else set()
            if not f.persisted or file_id not in owner_fsynced:
                reasons.append(
                    "persistence_boundary:%s(%s) 未在目录提交前完成 fsync"
                    % (name, file_id))
            if file_id not in self.manifest.digests:
                reasons.append("missing_digest:%s(%s)" % (name, file_id))
            elif f.digest is not None and f.digest != self.manifest.digests[file_id]:
                reasons.append("digest_mismatch:%s(%s)" % (name, file_id))
            # 交叉引用检查:文件必须属于本版本事务或某个已提交的旧版本
            owner_version = _txn_version_of(owner_txn)
            if f.txn_id != v.txn_id:
                if owner_version is None or owner_version not in committed_versions \
                        or owner_version >= v.version:
                    reasons.append(
                        "cross_version_reference:%s(%s) 属于事务 %s(版本 %s)"
                        % (name, file_id, f.txn_id, owner_version))
        return reasons

    def _check_wal(self, v):
        """从检查点 WAL 起点到版本 wal_end 必须连续覆盖。"""
        reasons = []
        cp = self.checkpoints.get(v.checkpoint_id)
        if cp is None:
            reasons.append("missing_checkpoint:%s" % v.checkpoint_id)
            return reasons
        if cp.wal_start > v.wal_start:
            reasons.append(
                "wal_start_before_checkpoint:版本 wal_start=%d 早于检查点 %d"
                % (v.wal_start, cp.wal_start))
        cursor = min(cp.wal_start, v.wal_start)
        segs = sorted(self.wal_segments, key=lambda s: s.start_lsn)
        while cursor < v.wal_end:
            nxt = None
            for s in segs:
                if s.start_lsn <= cursor < s.end_lsn:
                    if nxt is None or s.end_lsn > nxt.end_lsn:
                        nxt = s
            if nxt is None:
                reasons.append("missing_wal_segment:LSN %d 处无 WAL 段" % cursor)
                break
            cursor = nxt.end_lsn
        return reasons

    def _check_base_backup(self, v):
        reasons = []
        cp = self.checkpoints.get(v.checkpoint_id)
        if cp is None:
            return reasons  # 已由 _check_wal 报告
        if not cp.base_backup_id:
            reasons.append("missing_base_backup:检查点 %s 无基础备份" % cp.checkpoint_id)
            return reasons
        bb = self.base_backups.get(cp.base_backup_id)
        if bb is None:
            reasons.append("missing_base_backup:%s" % cp.base_backup_id)
        elif bb.catalog_version > v.version:
            reasons.append(
                "base_backup_too_new:备份对应版本 %d 晚于目录版本 %d"
                % (bb.catalog_version, v.version))
        return reasons

    # -- 整体验证 ---------------------------------------------------------

    def _committed_versions(self):
        out = set()
        for t in self.transactions.values():
            if t.status == "committed" and t.catalog_version is not None:
                out.add(t.catalog_version)
        return out

    def _version_reasons(self, v):
        reasons = []
        reasons += self._check_transaction(v)
        reasons += self._check_files(v)
        reasons += self._check_wal(v)
        reasons += self._check_base_backup(v)
        return reasons

    def validate(self):
        # type: () -> ValidationReport
        blocking = []
        status = {}
        for ver in sorted(self.versions):
            status[ver] = self._version_reasons(self.versions[ver])

        p = self.pointer
        target = None

        if p.primary != p.mirror:
            blocking.append(
                "pointer_switch_interrupted:主指针=%d 镜像指针=%d 不一致,"
                "目录切换在落盘途中被中断" % (p.primary, p.mirror))
            # 切换中断时,只有镜像(旧)一侧是双方都曾确认的状态
            target = p.mirror
        else:
            target = p.primary
            if target < p.high_water_mark:
                blocking.append(
                    "pointer_rollback:指针指向版本 %d,低于已提交高水位 %d,"
                    "疑似旧指针回退" % (target, p.high_water_mark))
                target = p.high_water_mark

        # 从目标版本向下找第一个通过全部检查、且事务已提交的版本
        recoverable = None
        if target is not None:
            candidates = sorted(
                (ver for ver in self.versions if ver <= target), reverse=True)
            committed = self._committed_versions()
            for ver in candidates:
                if not status[ver] and ver in committed:
                    recoverable = ver
                    break
                detail = "; ".join(status[ver]) if status[ver] else "事务未提交"
                blocking.append("version_%d_blocked:%s" % (ver, detail))
        if recoverable is None:
            blocking.append("no_recoverable_catalog:没有任何目录版本构成完整恢复点")

        orphaned = self._orphaned_objects(recoverable)
        return ValidationReport(
            recoverable_version=recoverable,
            orphaned_objects=orphaned,
            blocking_reasons=blocking,
            version_status=status,
        )

    def _orphaned_objects(self, recoverable):
        """已持久化且有摘要、但未被可恢复目录引用的文件。"""
        referenced = set()
        if recoverable is not None:
            referenced = set(self.versions[recoverable].entries.values())
        orphans = []
        for file_id in sorted(self.files):
            f = self.files[file_id]
            if f.persisted and file_id in self.manifest.digests \
                    and file_id not in referenced:
                orphans.append(file_id)
        return orphans
