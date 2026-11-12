"""Build full COCO train2017/val2017 valid person manifests.

Uses the existing supervision_filter_v2 rules.
Does not read, regenerate or modify the old train/dev split.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import fyp_data_common as filters


FILTER_VERSION = "supervision_filter_v2"
ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    """SHA-256 of the original file bytes, lowercase hexadecimal."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument(
        "--dataset-split",
        required=True,
        choices=["train2017", "val2017"],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "manifests",
    )
    args = parser.parse_args()

    expected_name = f"person_keypoints_{args.dataset_split}.json"
    if args.ann.name != expected_name:
        parser.error(
            f"--dataset-split {args.dataset_split} requires "
            f"annotation filename {expected_name}"
        )

    manifest_path = (
        args.out_dir / f"{args.dataset_split}_valid_persons.json"
    )
    stats_path = (
        args.out_dir / f"{args.dataset_split}_filter_stats.json"
    )

    for path in [manifest_path, stats_path]:
        if path.exists():
            parser.error(
                f"Output already exists: {path}. "
                "Choose a different --out-dir; no file was overwritten."
            )

    print("Reading:", args.ann)
    source_hash = file_sha256(args.ann)
    coco = filters.read_json(args.ann)

    images = {}
    for img in coco["images"]:
        image_id = img["id"]
        if image_id in images:
            raise ValueError(f"Duplicate image ID: {image_id}")
        images[image_id] = img

    valid_ids = []
    seen_ann_ids = set()
    valid_per_image = Counter()
    reason_counts = Counter()
    filtered_count = 0

    for ann in coco["annotations"]:
        ann_id = ann["id"]
        if type(ann_id) is not int:
            raise ValueError("Annotation IDs must be integers")
        if ann_id in seen_ann_ids:
            raise ValueError(f"Duplicate annotation ID: {ann_id}")
        seen_ann_ids.add(ann_id)

        image_id = ann["image_id"]
        if image_id not in images:
            raise ValueError(
                f"Annotation {ann_id} references unknown image {image_id}"
            )

        # Reuse the already-reviewed rules; no new private formula.
        reasons = filters.invalid_reasons(ann, images[image_id])

        if reasons:
            filtered_count += 1
            reason_counts.update(reasons)
        else:
            valid_ids.append(ann_id)
            valid_per_image[image_id] += 1

    valid_ids.sort()

    total_count = len(coco["annotations"])
    valid_count = len(valid_ids)

    if valid_count + filtered_count != total_count:
        raise AssertionError("Annotation counts do not balance")
    if len(valid_ids) != len(set(valid_ids)):
        raise AssertionError("Duplicate valid annotation IDs")

    # Make sure the source did not change while generating the results.
    if file_sha256(args.ann) != source_hash:
        raise RuntimeError(
            "Source annotation changed during processing; outputs not written"
        )

    manifest = {
        "dataset_split": args.dataset_split,
        "source_annotation": args.ann.name,
        "source_annotation_sha256": source_hash,
        "filter_version": FILTER_VERSION,
        "valid_ann_ids": valid_ids,
    }

    stats = {
        "dataset_split": args.dataset_split,
        "source_annotation": args.ann.name,
        "source_annotation_sha256": source_hash,
        "source_annotation_sha256_method": "sha256-raw-file-bytes",
        "filter_version": FILTER_VERSION,
        "filter_implementation": "scripts/fyp_data_common.py::invalid_reasons",
        "filter_implementation_sha256": file_sha256(
            Path(filters.__file__)
        ),
        "total_images": len(images),
        "annotations_before_filter": total_count,
        "valid_persons_after_filter": valid_count,
        "filtered_annotations": filtered_count,
        "images_with_valid_persons": len(valid_per_image),
        "images_without_valid_persons": len(images) - len(valid_per_image),
        "filter_reason_counts_nonexclusive": dict(
            sorted(reason_counts.items())
        ),
        "checks": {
            "valid_ids_are_integers": all(
                type(i) is int for i in valid_ids
            ),
            "valid_ids_unique": len(valid_ids) == len(set(valid_ids)),
            "valid_ids_sorted": valid_ids == sorted(valid_ids),
            "counts_balance": (
                valid_count + filtered_count == total_count
            ),
            "source_unchanged_during_generation": True,
            "old_90_10_split_used": False,
        },
        "notes": [
            "valid_ann_ids are person annotation IDs, not image IDs.",
            "The entire specified annotation file was processed.",
            "The original annotation file was not modified.",
            "v=1 occluded but labeled keypoints are retained.",
            "iscrowd filtering does not remove crowded-scene images.",
            "Filter reasons can overlap; do not sum reason counts.",
            "The original file-byte SHA-256 is not the old split checksum.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path, value in [
        (manifest_path, manifest),
        (stats_path, stats),
    ]:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print("\nDataset:", args.dataset_split)
    print("Images:", len(images))
    print("Annotations before filter:", total_count)
    print("Valid persons after filter:", valid_count)
    print("Filtered annotations:", filtered_count)
    print("Filter reasons:", dict(sorted(reason_counts.items())))
    print("Source annotation SHA-256:", source_hash)
    print("Saved:", manifest_path)
    print("Saved:", stats_path)
    print("PASS: full valid-person manifest generated.")


if __name__ == "__main__":
    main()