"""CPU-only tests for 2C storage, identity, resume and serialization contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mmpose/analysis/fyp"))
from shared_cache import (ReferenceCache, CacheWriter, SHAPES, INTEGER_FIELDS,
                          assert_disjoint, restore_coords)
from build_shared_cache import select_annotations, pack_batch


def batch(ids=(1, 2), images=(10, 20)):
    n = len(ids)
    a = {k: np.zeros((n,) + shape, dtype=np.int64 if k in INTEGER_FIELDS else np.float32)
         for k, shape in SHAPES.items()}
    a["annotation_id"][:] = ids
    a["image_id"][:] = images
    a["features"][:] = np.arange(17 * 256).reshape(17, 256) / 100
    a["confidence"][:] = 0.25
    a["uncertainty"][:] = 0.75
    a["bbox"][:] = [0, 0, 100, 200]
    a["input_size"][:] = [192, 256]
    a["input_scale"][:] = [100, 200]
    a["original_size"][:] = [640, 480]
    return a


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = {"dataset_split": "train2017", "checkpoint_sha256": "test",
                         "flip_test": False, "flip_protocol": "single_view"}

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, pairs=((10, 1), (20, 2)), name="cache"):
        return ReferenceCache.create(self.root / name, self.protocol, pairs)

    def test_roundtrip_and_no_pickle(self):
        c = self.cache()
        a = batch()
        with CacheWriter(c) as w:
            w.append(a)
        self.assertEqual(c.verify(True), {1, 2})
        loaded = next(c.iter_shards())
        for k in a:
            np.testing.assert_array_equal(a[k], loaded[k])

    def test_resume_partial_and_ignore_uncommitted_temporary(self):
        c = self.cache()
        with CacheWriter(c) as w:
            w.append(batch((1,), (10,)))
        (c.directory / "shard-999.npz.tmp").write_bytes(b"interrupted")
        with self.assertRaises(ValueError):
            c.verify(True)
        resumed = ReferenceCache.create(c.directory, self.protocol, [(10, 1), (20, 2)], True)
        # status is advisory: completed IDs are recovered from actual shards.
        (c.directory / "status.json").unlink()
        with CacheWriter(resumed) as w:
            self.assertEqual(w.seen, {1})
            w.append(batch((2,), (20,)))
        self.assertEqual(c.verify(True), {1, 2})

    def test_resume_rejects_config_or_ids(self):
        c = self.cache()
        with self.assertRaises(FileExistsError):
            ReferenceCache.create(c.directory, self.protocol, [(10, 1), (20, 2)])
        for p, ids in [(dict(self.protocol, flip_test=True), [(10, 1), (20, 2)]),
                       (self.protocol, [(99, 1), (20, 2)])]:
            with self.assertRaises(ValueError):
                ReferenceCache.create(c.directory, p, ids, True)

    def test_duplicate_and_wrong_identity_rejected(self):
        c = self.cache()
        with CacheWriter(c) as w:
            with self.assertRaises(ValueError):
                w.append(batch((1,), (999,)))
            w.append(batch((1,), (10,)))
            with self.assertRaises(ValueError):
                w.append(batch((1,), (10,)))

    def test_reject_bad_fields_shapes_nonfinite(self):
        c = self.cache()
        with CacheWriter(c) as w:
            a = batch(); a["gain"] = np.zeros(2)
            with self.assertRaises(ValueError): w.append(a)
            a = batch(); a["features"][0, 0, 0] = np.nan
            with self.assertRaises(ValueError): w.append(a)
            a = batch(); a["P_0_coords"] = np.zeros((2, 17, 3), dtype=np.float32)
            with self.assertRaises(ValueError): w.append(a)

    def test_foreign_shard_and_corruption_rejected(self):
        c = self.cache()
        with CacheWriter(c) as w: w.append(batch())
        path = c.shards()[0]
        a = batch()
        with path.open("wb") as f: np.savez(f, fingerprint=np.array("different-run"), **a)
        with self.assertRaises(ValueError): c.verify()
        path.write_bytes(b"not an npz")
        with self.assertRaises(ValueError): c.verify()

    def test_writer_lock(self):
        c = self.cache()
        with CacheWriter(c):
            with self.assertRaises(FileExistsError):
                with CacheWriter(c): pass
        self.assertFalse((c.directory / ".writer.lock").exists())

    def test_leakage_check(self):
        c = self.cache()
        with CacheWriter(c) as w: w.append(batch())
        v = ReferenceCache.create(self.root / "val", dict(self.protocol, dataset_split="val2017"), [(10, 3)])
        with CacheWriter(v) as w: w.append(batch((3,), (10,)))
        with self.assertRaisesRegex(ValueError, "image leakage"):
            assert_disjoint(c.directory, v.directory)

    def test_manifest_rejects_legacy_and_wrong_membership(self):
        coco = {"images": [{"id": 10}], "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "keypoints": [20, 30, 2]*17,
             "area": 100, "bbox": [0, 0, 10, 10], "num_keypoints": 17}]}
        manifest = {"dataset_split": "train2017", "source_annotation": "person_keypoints_train2017.json", "filter_version": "supervision_filter_v2", "valid_ann_ids": [1]}
        selected, _ = select_annotations(coco, manifest, "train2017", manifest["source_annotation"])
        self.assertEqual(selected[0]["id"], 1)
        with self.assertRaisesRegex(ValueError, "filter_version"):
            select_annotations(coco, dict(manifest, filter_version="legacy"), "train2017", manifest["source_annotation"])
        with self.assertRaises(ValueError):
            select_annotations(coco, {"train_valid_ann_ids": [1]}, "train2017", manifest["source_annotation"])
        with self.assertRaises(ValueError):
            select_annotations(coco, dict(manifest, valid_ann_ids=[2]), "train2017", manifest["source_annotation"])
        with self.assertRaises(ValueError):
            select_annotations(coco, manifest, "val2017", manifest["source_annotation"])

    def test_original_coordinate_restoration(self):
        coords = np.array([[96, 128], [0, 0]], dtype=np.float32)
        restored = restore_coords(coords, np.array([200, 300]), np.array([100, 200]), np.array([192, 256]))
        np.testing.assert_allclose(restored, [[200, 300], [150, 200]])

    def test_metadata_identity_required_not_loader_order(self):
        from types import SimpleNamespace
        result = SimpleNamespace(metainfo={}, pred_instances=None)
        state = {k: np.zeros((1,)) for k in
                 ("features", "uncertainty", "confidence", "top1_indices")}
        with self.assertRaisesRegex(ValueError, "Missing annotation/image ID"):
            pack_batch(state, [result], {1: 10}, {})

    def test_pack_reordered_people_keeps_scores_and_identity(self):
        from types import SimpleNamespace
        a = batch((2, 1), (20, 10))
        state = {k: a[k] for k in ("features", "uncertainty", "confidence")}
        state["top1_indices"] = a["top1_joint"]
        results = []
        for b in range(2):
            meta = {"id": int(a["annotation_id"][b]), "img_id": int(a["image_id"][b]),
                    "input_center": a["input_center"][b], "input_scale": a["input_scale"][b],
                    "input_size": a["input_size"][b]}
            pred = SimpleNamespace(keypoints=np.full((1,17,2), b+10, dtype=np.float32),
                                   keypoint_scores=np.full((1,17), 0.2+b*0.5, dtype=np.float32),
                                   bboxes=a["bbox"][b:b+1], bbox_scores=np.ones(1, dtype=np.float32))
            results.append(SimpleNamespace(metainfo=meta, pred_instances=pred))
        images = {i: {"width":640,"height":480} for i in (10,20)}
        packed = pack_batch(state, results, {1:10,2:20}, images)
        np.testing.assert_array_equal(packed["annotation_id"], [2,1])
        np.testing.assert_allclose(packed["P_0_scores"][:,0], [0.2,0.7])
        with self.assertRaisesRegex(ValueError, "does not match"):
            pack_batch(state, results, {1:20,2:10}, images)

    def test_duplicate_shards_detected_even_if_status_says_complete(self):
        import shutil
        c = self.cache()
        with CacheWriter(c) as w: w.append(batch())
        shutil.copyfile(c.shards()[0], c.directory / "shard-999999999999.npz")
        with self.assertRaisesRegex(ValueError, "across shards"):
            c.verify(True)

    def test_disjoint_full_caches_pass(self):
        c = self.cache()
        with CacheWriter(c) as w: w.append(batch())
        v = ReferenceCache.create(self.root / "val", dict(self.protocol, dataset_split="val2017"), [(30,3)])
        with CacheWriter(v) as w: w.append(batch((3,), (30,)))
        assert_disjoint(c.directory, v.directory)


if __name__ == "__main__":
    unittest.main()
