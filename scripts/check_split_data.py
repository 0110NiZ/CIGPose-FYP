"""Recheck an EXISTING split; never regenerate or edit its image IDs."""
import argparse
import json
from pathlib import Path
from fyp_data_common import audit_coco, check_outputs, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--split", type=Path, default=ROOT / "splits/train_dev_split.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/data_check")
    parser.add_argument("--overwrite", action="store_true", help="Replace reports only, never the split")
    args = parser.parse_args()
    report_path = args.out_dir / "train_dev_data_check_v2.json"
    manifest_path = args.out_dir / "train_dev_valid_ann_ids_v2.json"
    if args.split.resolve() in {report_path.resolve(), manifest_path.resolve()}:
        parser.error("Output must not replace the split")
    check_outputs([report_path, manifest_path], args.overwrite)
    original = args.split.read_bytes()
    split = read_json(args.split)
    if split["source_annotation"] != args.ann.name:
        parser.error("Annotation basename differs from the recorded split source")
    report, manifest = audit_coco(read_json(args.ann), split)
    report["split_file"] = args.split.name
    if args.split.read_bytes() != original:
        raise RuntimeError("Split was changed during checking")
    write_json(report_path, report)
    write_json(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", report_path)
    print("Saved:", manifest_path)
    print("PASS: existing image split unchanged.")


if __name__ == "__main__":
    main()
