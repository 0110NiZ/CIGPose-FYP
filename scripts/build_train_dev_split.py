"""Generate a NEW image split only. Reuse the committed split for this project."""
import argparse
import random
from pathlib import Path
from fyp_data_common import audit_coco, check_outputs, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must lie between 0 and 1")
    paths = [args.out_dir / name for name in (
        "train_dev_split.json", "train_dev_data_check_v2.json",
        "train_dev_valid_ann_ids_v2.json")]
    # There is deliberately no overwrite option for split generation.
    check_outputs(paths)
    coco = read_json(args.ann)
    ids = sorted(int(img["id"]) for img in coco["images"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source image IDs")
    random.Random(args.seed).shuffle(ids)
    n = int(len(ids) * args.train_ratio)
    split = {
        "version": "coco_train2017_image_split_v1",
        "source_annotation": args.ann.name,
        "seed": args.seed, "train_ratio": args.train_ratio,
        "split_unit": "image_id",
        "image_scope": "all images listed in the source annotation",
        "train_image_ids": sorted(ids[:n]), "dev_image_ids": sorted(ids[n:]),
    }
    report, manifest = audit_coco(coco, split)
    report["split_file"] = paths[0].name
    for path, value in zip(paths, [split, report, manifest]):
        write_json(path, value)
        print("Saved:", path)
    print("PASS: new split generated. Do not replace an existing team split.")


if __name__ == "__main__":
    main()
