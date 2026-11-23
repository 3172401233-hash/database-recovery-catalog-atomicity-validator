"""catalog_validator 的单元测试。

覆盖:目录切换中断、双版本交叉引用、WAL 缺失、旧指针回退,
以及部分写入、摘要缺失、孤立对象识别和正常恢复路径。
"""

import copy
import unittest

from catalog_validator import (
    BaseBackup,
    CatalogAtomicityValidator,
    CatalogVersion,
    Checkpoint,
    DigestManifest,
    DualPointer,
    FileRecord,
    UpdateTransaction,
    WALSegment,
)


def make_world():
    """构造一个完全一致的模拟世界:v2 是最新可恢复版本。"""
    files = [
        FileRecord("f1", txn_id="t1", persisted=True, digest="d1"),
        FileRecord("f2", txn_id="t2", persisted=True, digest="d2"),
    ]
    versions = [
        CatalogVersion(version=1, txn_id="t1", entries={"a": "f1"},
                       checkpoint_id="cp1", wal_start=0, wal_end=100),
        CatalogVersion(version=2, txn_id="t2", entries={"a": "f1", "b": "f2"},
                       checkpoint_id="cp1", wal_start=0, wal_end=200),
    ]
    pointer = DualPointer(primary=2, mirror=2, high_water_mark=2)
    checkpoints = [Checkpoint("cp1", wal_start=0, base_backup_id="bb1")]
    wal = [WALSegment("seg0", 0, 100), WALSegment("seg1", 100, 200)]
    backups = [BaseBackup("bb1", catalog_version=1, checkpoint_id="cp1")]
    manifest = DigestManifest({"f1": "d1", "f2": "d2"})
    txns = [
        UpdateTransaction("t1", "committed", fsynced_files=["f1"],
                          catalog_version=1),
        UpdateTransaction("t2", "committed", fsynced_files=["f2"],
                          catalog_version=2),
    ]
    return dict(catalog_versions=versions, pointer=pointer,
                checkpoints=checkpoints, wal_segments=wal,
                base_backups=backups, manifest=manifest,
                transactions=txns, files=files)


def validate(world):
    return CatalogAtomicityValidator(**world).validate()


class CleanRecoveryTest(unittest.TestCase):
    def test_clean_recovery_point(self):
        report = validate(make_world())
        self.assertEqual(report.recoverable_version, 2)
        self.assertEqual(report.orphaned_objects, [])
        self.assertEqual(report.blocking_reasons, [])
        self.assertTrue(report.recoverable)


class SwitchInterruptedTest(unittest.TestCase):
    def test_catalog_switch_interrupted(self):
        world = make_world()
        # 崩溃发生在指针切换途中:主指针已指向 v2,镜像仍是 v1
        world["pointer"] = DualPointer(primary=2, mirror=1, high_water_mark=2)
        report = validate(world)
        self.assertTrue(any("pointer_switch_interrupted" in r
                            for r in report.blocking_reasons))
        # 只能回到双方都确认过的 v1
        self.assertEqual(report.recoverable_version, 1)
        # f2 已持久化但不被 v1 引用 -> 孤立对象
        self.assertEqual(report.orphaned_objects, ["f2"])


class CrossVersionReferenceTest(unittest.TestCase):
    def test_dual_version_cross_reference(self):
        world = make_world()
        # v2 的一条记录指向由 v3 事务写入的文件:单独看完全合法
        world["files"].append(
            FileRecord("f3", txn_id="t3", persisted=True, digest="d3"))
        world["manifest"].digests["f3"] = "d3"
        world["transactions"].append(
            UpdateTransaction("t3", "committed", fsynced_files=["f3"],
                              catalog_version=3))
        world["catalog_versions"][1].entries["b"] = "f3"
        report = validate(world)
        self.assertTrue(any("cross_version_reference" in r
                            for r in report.version_status[2]))
        self.assertEqual(report.recoverable_version, 1)


class WalMissingTest(unittest.TestCase):
    def test_wal_segment_missing(self):
        world = make_world()
        # 丢掉 [100,200) 段,v2 的 WAL 链断裂
        world["wal_segments"] = [WALSegment("seg0", 0, 100)]
        report = validate(world)
        self.assertTrue(any("missing_wal_segment" in r
                            for r in report.version_status[2]))
        self.assertEqual(report.recoverable_version, 1)

    def test_wal_gap_in_middle(self):
        world = make_world()
        world["catalog_versions"][1].wal_end = 300
        world["wal_segments"] = [WALSegment("seg0", 0, 100),
                                 WALSegment("seg2", 200, 300)]
        report = validate(world)
        self.assertTrue(any("missing_wal_segment" in r
                            for r in report.version_status[2]))
        self.assertEqual(report.recoverable_version, 1)


class PointerRollbackTest(unittest.TestCase):
    def test_old_pointer_rollback(self):
        world = make_world()
        # 指针被旧副本覆盖回 v1,但高水位记录 v2 已提交过
        world["pointer"] = DualPointer(primary=1, mirror=1, high_water_mark=2)
        report = validate(world)
        self.assertTrue(any("pointer_rollback" in r
                            for r in report.blocking_reasons))
        # 回退不被接受,恢复到高水位对应的有效版本
        self.assertEqual(report.recoverable_version, 2)


class PartialWriteTest(unittest.TestCase):
    def test_partial_write_blocks_new_version(self):
        world = make_world()
        # t2 写到一半崩溃:f2 未 fsync,事务状态 partial
        world["transactions"][1] = UpdateTransaction(
            "t2", "partial", fsynced_files=[], catalog_version=2)
        world["files"][1].persisted = False
        report = validate(world)
        reasons = report.version_status[2]
        self.assertTrue(any("partial_write" in r for r in reasons))
        self.assertTrue(any("persistence_boundary" in r for r in reasons))
        self.assertEqual(report.recoverable_version, 1)


class MissingDigestTest(unittest.TestCase):
    def test_missing_digest_blocks_version(self):
        world = make_world()
        del world["manifest"].digests["f2"]
        report = validate(world)
        self.assertTrue(any("missing_digest" in r
                            for r in report.version_status[2]))
        self.assertEqual(report.recoverable_version, 1)


class OrphanedObjectsTest(unittest.TestCase):
    def test_orphaned_objects_reported(self):
        world = make_world()
        world["files"].append(
            FileRecord("f9", txn_id="t1", persisted=True, digest="d9"))
        world["manifest"].digests["f9"] = "d9"
        report = validate(world)
        self.assertEqual(report.recoverable_version, 2)
        self.assertEqual(report.orphaned_objects, ["f9"])

    def test_unpersisted_file_is_not_orphan(self):
        world = make_world()
        world["files"].append(
            FileRecord("f8", txn_id="t1", persisted=False, digest=None))
        report = validate(world)
        self.assertEqual(report.orphaned_objects, [])


class NoRecoverableTest(unittest.TestCase):
    def test_no_recoverable_catalog(self):
        world = make_world()
        # 两个版本都缺摘要
        world["manifest"] = DigestManifest({})
        report = validate(world)
        self.assertIsNone(report.recoverable_version)
        self.assertFalse(report.recoverable)
        self.assertTrue(any("no_recoverable_catalog" in r
                            for r in report.blocking_reasons))


if __name__ == "__main__":
    unittest.main()
