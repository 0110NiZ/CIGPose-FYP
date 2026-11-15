
"""Step 3: Cache integration and Train/Val split validation.

Usage:
    python step3_validation.py \
        --coco-root PATH \
        --manifest-dir PATH \
        --smoke-dir PATH \
        --generation-log PATH

This script audits an existing 16-person cache and its original
live-inference/reload log. Full cache auditing belongs to Step 5.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]

FIELDS = {
    "features": (17, 256),
    "uncertainty": (17,),
    "confidence": (17,),
    "P_0_coords": (17, 2),
    "P_0_scores": (17,),
    "bbox": (4,),
    "bbox_score": (),
    "input_center": (2,),
    "input_scale": (2,),
    "input_size": (2,),
    "original_size": (2,),
    "top1_joint": (),
    "image_id": (),
    "annotation_id": ()
}

FLOATS = set(FIELDS) - {
    "image_id",
    "annotation_id",
    "top1_joint"
}


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig")
    )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def git_head():
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True
    )

    return (
        result.stdout.strip()
        if result.returncode == 0
        else "unavailable"
    )


def audit_split(
    name, expected_count,
    manifest_dir, coco_root, errors
):
    manifest_path = (
        manifest_dir
        / f"{name}_valid_persons.json"
    )

    gt_path = (
        coco_root
        / "annotations"
        / f"person_keypoints_{name}.json"
    )

    manifest = load_json(manifest_path)
    gt = load_json(gt_path)

    ids = [
        int(x) for x in manifest["valid_ann_ids"]
    ]

    unique_ids = set(ids)

    annotations = {
        int(a["id"]): a
        for a in gt["annotations"]
    }

    image_ids = {
        int(image["id"])
        for image in gt["images"]
    }

    images_selected = {
        int(annotations[aid]["image_id"])
        for aid in unique_ids
        if aid in annotations
    }

    hash_ok = (
        sha256(gt_path)
        == manifest.get("source_annotation_sha256")
    )

    if not hash_ok:
        errors.append(
            f"{name}: annotation SHA256 mismatch"
        )

    if (
        len(ids) != expected_count
        or len(ids) != len(unique_ids)
    ):
        errors.append(
            f"{name}: count or uniqueness mismatch"
        )

    if (
        len(annotations)
        != len(gt["annotations"])
    ):
        errors.append(
            f"{name}: duplicate GT annotation IDs"
        )

    if not unique_ids.issubset(annotations):
        errors.append(
            f"{name}: annotations missing in COCO GT"
        )

    if not images_selected.issubset(image_ids):
        errors.append(
            f"{name}: image IDs not in COCO GT"
        )

    expected_fields = [
        ("dataset_split", name),
        ("source_annotation", gt_path.name),
        ("filter_version", "supervision_filter_v2")
    ]

    for field, expected in expected_fields:
        if manifest.get(field) != expected:
            errors.append(
                f"{name}: {field} mismatch"
            )

    # Verify the validity of selected COCO annotations.
    for aid in unique_ids:
        ann = annotations.get(aid)

        if ann is None:
            continue

        keypoints = np.asarray(
            ann["keypoints"]
        )

        if (
            ann.get("category_id") != 1
            or ann.get("iscrowd", 0)
            or ann.get("area", 0) <= 0
            or keypoints.shape != (51,)
            or not np.isin(
                keypoints[2::3], [0, 1, 2]
            ).all()
            or not (keypoints[2::3] > 0).any()
        ):
            errors.append(
                f"{name}: invalid annotation {aid}"
            )
            break

    print(
        name,
        "persons:", len(ids),
        "images:", len(images_selected),
        "SHA256:",
        "PASS" if hash_ok else "FAIL"
    )

    return {
        "ids": unique_ids,
        "images": images_selected,
        "all_images": image_ids,
        "by_id": annotations,
        "count": len(ids),
        "hash_ok": hash_ok
    }


def audit_smoke(smoke_dir, val, errors):
    shards = sorted(
        smoke_dir.glob("shard-*.npz")
    )

    if not shards:
        raise FileNotFoundError(
            f"No smoke-cache NPZ files: {smoke_dir}"
        )

    status = load_json(
        smoke_dir / "status.json"
    )

    selected = load_json(
        smoke_dir / "selected_annotations.json"
    )

    selected_annotations = {
        int(a["id"]): a
        for a in selected["annotations"]
    }

    pairs = []
    fingerprints = set()

    for shard in shards:
        with np.load(
            shard, allow_pickle=False
        ) as data:

            missing = (
                set(FIELDS) - set(data.files)
            )

            if missing:
                errors.append(
                    f"{shard.name}: "
                    f"missing fields {sorted(missing)}"
                )
                continue

            if "fingerprint" not in data.files:
                errors.append(
                    f"{shard.name}: missing fingerprint"
                )
            else:
                fingerprints.add(
                    str(data["fingerprint"].item())
                )

            n = len(data["annotation_id"])

            # Check field shapes and numeric validity.
            for field, shape in FIELDS.items():
                if (
                    data[field].shape
                    != (n,) + shape
                ):
                    errors.append(
                        f"{shard.name}: "
                        f"invalid {field} shape"
                    )

                if (
                    field in FLOATS
                    and not np.isfinite(
                        data[field]
                    ).all()
                ):
                    errors.append(
                        f"{shard.name}: "
                        f"nonfinite {field}"
                    )

            # Check the identity of each cached person.
            for image_id, ann_id, joint in zip(
                data["image_id"],
                data["annotation_id"],
                data["top1_joint"]
            ):
                image_id = int(image_id)
                ann_id = int(ann_id)
                joint = int(joint)

                pairs.append(
                    (image_id, ann_id)
                )

                gt_ann = val["by_id"].get(
                    ann_id
                )

                if (
                    gt_ann is None
                    or ann_id not in val["ids"]
                    or int(gt_ann["image_id"])
                    != image_id
                ):
                    errors.append(
                        "Smoke: GT identity mismatch "
                        f"for {ann_id}"
                    )

                selected_ann = (
                    selected_annotations.get(ann_id)
                )

                if (
                    selected_ann is None
                    or int(selected_ann["image_id"])
                    != image_id
                ):
                    errors.append(
                        "Smoke: selected annotation "
                        f"mismatch for {ann_id}"
                    )

                if not 0 <= joint < 17:
                    errors.append(
                        f"Smoke: invalid joint {joint}"
                    )

    # Check cache count and uniqueness.
    if (
        len(pairs) != 16
        or len(set(pairs)) != 16
    ):
        errors.append(
            "Smoke: expected 16 unique instances"
        )

    # Check fingerprints across all shards.
    if (
        len(fingerprints) != 1
        or status.get("fingerprint")
        not in fingerprints
    ):
        errors.append(
            "Smoke: fingerprint mismatch"
        )

    if (
        status.get("count") != len(pairs)
        or status.get("expected") != len(pairs)
        or status.get("complete") is not True
    ):
        errors.append(
            "Smoke: status count/completeness mismatch"
        )

    if (
        set(selected_annotations)
        != {ann_id for _, ann_id in pairs}
    ):
        errors.append(
            "Smoke: selected annotations mismatch"
        )

    return {
        "shards": len(shards),
        "instances": len(pairs),
        "unique_pairs": len(set(pairs)),
        "fingerprints": len(fingerprints),
        "complete": status.get("complete")
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--coco-root",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--smoke-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--generation-log",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "validation_results/step3"
    )

    args = parser.parse_args()
    args.out_dir.mkdir(
        parents=True, exist_ok=True
    )

    errors = []

    report = {
        "task": "Step 3 Integration Validation",
        "git_commit": git_head(),
        "overall": "FAIL"
    }

    try:
        # Full Train/Val manifest integrity checks.
        train = audit_split(
            "train2017",
            149813,
            args.manifest_dir,
            args.coco_root,
            errors
        )

        val = audit_split(
            "val2017",
            6352,
            args.manifest_dir,
            args.coco_root,
            errors
        )

        # Verify that Train and Val remain separate.
        person_overlap = len(
            train["ids"] & val["ids"]
        )

        image_overlap = len(
            train["images"] & val["images"]
        )

        source_image_overlap = len(
            train["all_images"]
            & val["all_images"]
        )

        if (
            person_overlap
            or image_overlap
            or source_image_overlap
        ):
            errors.append(
                "Train/Val annotation or image overlap"
            )

        # Independently audit the existing 16-person cache.
        smoke = audit_smoke(
            args.smoke_dir,
            val,
            errors
        )

        # Verify the original live-model integration log.
        original_log = (
            args.generation_log.read_text(
                encoding="utf-8",
                errors="replace"
            )
        )

        required_markers = [
            "PASS: reloaded P0 and mixed-action "
            "candidate match live path",
            "PASS: smoke cache complete (16 instances)"
        ]

        log_ok = all(
            marker in original_log
            for marker in required_markers
        )

        shutil.copy2(
            args.generation_log,
            args.out_dir / "step3_generation.log"
        )

        if not log_ok:
            errors.append(
                "Original real-model/reload "
                "test log does not show PASS"
            )

        report.update({
            "train_persons": train["count"],
            "val_persons": val["count"],
            "train_gt_sha256_match": (
                train["hash_ok"]
            ),
            "val_gt_sha256_match": (
                val["hash_ok"]
            ),
            "train_val_person_overlap": (
                person_overlap
            ),
            "train_val_image_overlap": (
                image_overlap
            ),
            "train_val_source_image_overlap": (
                source_image_overlap
            ),
            "smoke_shards": smoke["shards"],
            "smoke_instances": smoke["instances"],
            "smoke_unique_pairs": (
                smoke["unique_pairs"]
            ),
            "smoke_fingerprints": (
                smoke["fingerprints"]
            ),
            "smoke_status_complete": (
                smoke["complete"]
            ),
            "real_model_generation_log_pass": (
                log_ok
            )
        })

        report["overall"] = (
            "PASS" if not errors else "FAIL"
        )

    except Exception as exc:
        errors.append(
            f"{type(exc).__name__}: {exc}"
        )

    report["error_count"] = len(errors)
    report["errors"] = errors[:50]

    report_path = (
        args.out_dir / "step3_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    (
        args.out_dir / "step3_run.log"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSTEP 3 FINAL SUMMARY")

    for key, value in report.items():
        if key not in (
            "errors",
            "git_commit"
        ):
            print(f"{key}: {value}")

    print("Results:", args.out_dir)

    return (
        0 if report["overall"] == "PASS" else 1
    )


if __name__ == "__main__":
    sys.exit(main())
