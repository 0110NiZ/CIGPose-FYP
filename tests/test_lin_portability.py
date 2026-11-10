"""Portable CLI regression tests; fixtures are synthetic, not COCO experiments."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fyp_data_common import (SPLIT_HASH_METHOD, audit_coco, hash_metadata,
                             read_json, split_sha256, verify_manifest, write_json)
from migrate_split_checksums import migrate


def fixture():
    images = [{"id": i, "width": 200, "height": 200} for i in range(1, 11)]
    anns = []
    for i in range(1, 11):
        anns.append({"id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
                     "bbox": [10, 10, 150, 150], "area": 20000,
                     "num_keypoints": 17,
                     "keypoints": [v for j in range(17) for v in (50 + j, 60 + j, 1 if j == 0 else 2)]})
    for j, updates in enumerate([
        {"iscrowd": 1}, {"keypoints": [0] * 51, "num_keypoints": 0},
        {"area": 0}, {"bbox": [0, 0, 0, 10]},
        {"bbox": [300, 300, 10, 10]},
    ], 11):
        a = copy.deepcopy(anns[0])
        a.update(updates, id=j)
        anns.append(a)
    split = {"source_annotation": "person_keypoints_train2017.json", "seed": 42,
             "train_image_ids": list(range(1, 6)), "dev_image_ids": list(range(6, 11))}
    return {"images": images, "annotations": anns}, split


class PortabilityTests(unittest.TestCase):
    def run_cli(self, script, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=ROOT, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_line_endings_formatting_and_bom(self):
        _, split = fixture()
        with tempfile.TemporaryDirectory() as d:
            for i, raw in enumerate([
                json.dumps(split, indent=2).encode(),
                json.dumps(split, indent=2).replace("\n", "\r\n").encode(),
                json.dumps(dict(reversed(list(split.items()))), separators=(",", ":")).encode(),
                b"\xef\xbb\xbf" + json.dumps(split).encode(),
            ]):
                p = Path(d) / str(i)
                p.write_bytes(raw)
                self.assertEqual(split_sha256(read_json(p)), split_sha256(split))

    def test_membership_change_rejected(self):
        _, split = fixture()
        manifest = hash_metadata(split)
        modified = copy.deepcopy(split)
        modified["dev_image_ids"][0] = 999
        with self.assertRaises(ValueError):
            verify_manifest(modified, manifest)
        with self.assertRaises(ValueError):
            verify_manifest(split, {"split_sha256": split_sha256(split)})

    def test_existing_full_split_and_manifest(self):
        split = read_json(ROOT / "splits/train_dev_split.json")
        manifest = read_json(ROOT / "splits/train_dev_valid_ann_ids_v2.json")
        verify_manifest(split, manifest)
        self.assertEqual(len(split["train_image_ids"]), 106458)
        self.assertEqual(len(split["dev_image_ids"]), 11829)
        self.assertFalse(set(split["train_image_ids"]) & set(split["dev_image_ids"]))
        self.assertEqual(len(manifest["train_valid_ann_ids"]), 134701)
        self.assertEqual(len(manifest["dev_valid_ann_ids"]), 15112)

    def test_filter_and_statistics(self):
        coco, split = fixture()
        report, manifest = audit_coco(coco, split)
        self.assertEqual(manifest["train_valid_ann_ids"], list(range(1, 6)))
        self.assertEqual(manifest["dev_valid_ann_ids"], list(range(6, 11)))
        self.assertEqual(report["statistics"]["train"]["filtered_persons"], 5)
        self.assertAlmostEqual(report["statistics"]["dev"]["occluded_fraction_among_labeled_keypoints"], 1 / 17)

    def test_check_cli_separate_output_and_no_overwrite(self):
        coco, split = fixture()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ann, sp = d / split["source_annotation"], d / "input.json"
            write_json(ann, coco)
            sp.write_bytes(json.dumps(split, indent=2).replace("\n", "\r\n").encode())
            before = sp.read_bytes()
            args = ("--ann", ann, "--split", sp, "--out-dir", d / "check")
            self.run_cli("scripts/check_split_data.py", *args)
            self.assertEqual(sp.read_bytes(), before)
            self.run_cli("scripts/check_split_data.py", *args, success=False)
            self.run_cli("scripts/check_split_data.py", *args, "--overwrite")
            self.run_cli("scripts/check_split_data.py", "--ann", ann, "--split", sp,
                         "--out-dir", d / "independent")

    def test_generate_cli_and_freeze(self):
        coco, _ = fixture()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ann = d / "person_keypoints_train2017.json"
            write_json(ann, coco)
            args = ("--ann", ann, "--out-dir", d / "generated", "--seed", 42, "--train-ratio", 0.5)
            self.run_cli("scripts/build_train_dev_split.py", *args)
            generated = read_json(d / "generated/train_dev_split.json")
            self.assertEqual(len(generated["train_image_ids"]), 5)
            self.run_cli("scripts/build_train_dev_split.py", *args, success=False)

    def test_migration_verifies_legacy_and_preserves_results(self):
        _, split = fixture()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sp = d / "split.json"
            raw = json.dumps(split, indent=2).replace("\n", "\r\n").encode()
            sp.write_bytes(raw)
            old_hash = hashlib.sha256(raw).hexdigest()
            values = [{"split_sha256": old_hash, "keep": [1, 2, 3]} for _ in range(3)]
            paths = [d / f"input{i}.json" for i in range(3)]
            for p, value in zip(paths, values):
                write_json(p, value)
            metadata, outputs = migrate(sp, *paths, d / "out")
            self.assertEqual(metadata["split_hash_method"], SPLIT_HASH_METHOD)
            self.assertEqual(sp.read_bytes(), raw)
            for output in outputs:
                migrated = read_json(output)
                self.assertEqual(migrated["keep"], [1, 2, 3])
                self.assertFalse(migrated["checksum_metadata_migration"]["results_recomputed"])
            bad = read_json(paths[0])
            bad["split_sha256"] = "0" * 64
            write_json(paths[0], bad)
            with self.assertRaises(ValueError):
                migrate(sp, *paths, d / "bad")

    def test_official_oks_cli_with_synthetic_fixture(self):
        coco, split = fixture()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ann, sp, mp = d / split["source_annotation"], d / "split.json", d / "manifest.json"
            write_json(ann, coco)
            write_json(sp, split)
            _, manifest = audit_coco(coco, split)
            write_json(mp, manifest)
            args = ("--ann", ann, "--split", sp, "--manifest", mp, "--out-dir", d / "oks")
            result = self.run_cli("mmpose/analysis/fyp/check_oks.py", *args)
            self.assertIn("ALL CHECKS PASSED", result.stdout)
            report = read_json(d / "oks/oks_validation.json")
            self.assertEqual(len(report["official_comparisons"]), 19)
            self.assertEqual(report["max_absolute_difference"], 0)
            self.run_cli("mmpose/analysis/fyp/check_oks.py", *args, success=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
