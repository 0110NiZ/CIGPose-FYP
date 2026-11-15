"""Synthetic input regression tests, not a model-inference experiment."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from read_val_detection_boxes import (_load, detection_instance_id, export_delivery,
                                      load_detection_instances, select_samples)


class DetectionReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.det = self.root / "detections.json"
        self.ann = self.root / "person_keypoints_val2017.json"
        self.boxes = [
            {"image_id": 7, "category_id": 1, "bbox": [-2, 3, 10, 20], "score": 0.0},
            {"image_id": 7, "category_id": 1, "bbox": [-2, 3, 10, 20], "score": 0.9},
            {"image_id": 9, "category_id": 1, "bbox": [5, 6, 20, 30], "score": 0.1},
        ]
        # No person annotations at all: GT person lists must not filter detections.
        self.ann.write_text(json.dumps({"images": [
            {"id": 7, "file_name": "000000000007.jpg", "width": 100, "height": 100},
            {"id": 9, "file_name": "000000000009.jpg", "width": 100, "height": 100}],
            "annotations": []}), encoding="utf-8")

    def load(self, boxes=None, image_root=None):
        self.det.write_text(json.dumps(self.boxes if boxes is None else boxes), encoding="utf-8")
        raw = self.det.read_bytes()
        return _load(self.det, self.ann, image_root, hashlib.sha256(raw).hexdigest(),
                     len(self.boxes if boxes is None else boxes))

    def test_zero_score_overlap_and_no_gt_filter(self):
        meta, records = self.load()
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["bbox_score"], 0.0)
        self.assertFalse(meta["additional_bbox_nms"])
        self.assertFalse(meta["gt_person_list_used_for_filtering"])

    def test_stable_identity_and_source_order(self):
        _, a = self.load()
        _, b = self.load()
        self.assertEqual(a, b)
        self.assertEqual([r["source_detection_index"] for r in a], [0, 1, 2])
        self.assertEqual(len({r["instance_id"] for r in a}), 3)
        self.assertEqual(a[0]["instance_id"], a[0]["detection_instance_id"])

    def test_box_and_score_are_preserved(self):
        _, records = self.load()
        for raw, record in zip(self.boxes, records):
            self.assertEqual(raw["bbox"], record["bbox"])
            self.assertEqual(raw["score"], record["bbox_score"])
        self.assertEqual(records[0]["bbox_xyxy"], [-2, 3, 8, 23])
        self.assertNotIn("annotation_id", records[0])

    def test_unknown_source_is_rejected(self):
        self.load()
        with self.assertRaises(ValueError):
            load_detection_instances(self.det, self.ann)

    def test_bad_records_fail_instead_of_being_dropped(self):
        for update in [{"image_id": 100}, {"category_id": 2},
                       {"bbox": [0, 0, 0, 4]}, {"score": float("nan")}]:
            boxes = [dict(self.boxes[0], **update)]
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.load(boxes)

    def test_sampling_does_not_change_full_list(self):
        _, records = self.load()
        original = json.dumps(records)
        samples = select_samples(records, 2, 1)
        self.assertEqual(len(samples), 2)
        self.assertEqual([r["source_detection_index"] for r in samples], [0, 2])
        self.assertEqual(json.dumps(records), original)

    def test_export_identity_and_overwrite_protection(self):
        meta, records = self.load()
        out = self.root / "result"
        report, paths = export_delivery(meta, records, out)
        manifest = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["detection_instance_ids"], [r["instance_id"] for r in records])
        self.assertTrue(report["checks"]["all_source_records_retained"])
        self.assertFalse(report["real_model_forward_checked"])
        with self.assertRaises(FileExistsError):
            export_delivery(meta, records, out)

    def test_sample_image_paths_and_missing_images(self):
        meta, records = self.load(image_root=self.root / "val2017")
        with self.assertRaises(FileNotFoundError):
            export_delivery(meta, records, self.root / "missing", image_root=self.root / "val2017")
        (self.root / "val2017").mkdir()
        for name in ["000000000007.jpg", "000000000009.jpg"]:
            (self.root / "val2017" / name).write_bytes(b"fixture")
        report, _ = export_delivery(meta, records, self.root / "found", image_root=self.root / "val2017")
        self.assertTrue(report["sample_image_files_checked"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
