
"""Step 5: Independent full shared-cache validation.

Usage:
    python step5_validation.py --cache-source PATH \
        --coco-root PATH --manifest-dir PATH

The cache source can be an extracted directory or a ZIP.
All cache files are read-only. No GPU or model inference is required.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
import zipfile

import numpy as np


REPO = Path(__file__).resolve().parents[3]

SHAPES = {
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

INT_FIELDS = {
    "top1_joint", "image_id", "annotation_id"
}

COUNTS = {
    "train2017": 149813,
    "val2017": 6352
}

SCHEMA = "cigpose-reference-cache-v1"

CHECKPOINT_SHA256 = (
    "3dd69c8ab4e3c7a923a0183fd9abf844"
    "cb7ef13d39d26693baa082ca7931b3d0"
)


def digest_json(obj):
    content = json.dumps(
        obj, sort_keys=True,
        separators=(",", ":"),
        allow_nan=False
    )
    return hashlib.sha256(content.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig")
    )


class CacheSource:
    """Read NPZ shards sequentially from a directory or ZIP."""

    def __init__(self, source):
        self.source = Path(source)
        self.zf = None
        self.locations = {}

        if (
            self.source.is_file()
            and zipfile.is_zipfile(self.source)
        ):
            self.zf = zipfile.ZipFile(self.source)
            names = self.zf.namelist()

            for split in COUNTS:
                suffix = f"/{split}/run.json"

                matches = [
                    n for n in names
                    if n.endswith(suffix)
                    or n == f"{split}/run.json"
                ]

                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one {split}/run.json "
                        f"in ZIP; found {len(matches)}"
                    )

                self.locations[split] = (
                    matches[0][:-len("run.json")]
                )

            self.names = set(names)

        elif self.source.is_dir():
            for split in COUNTS:
                matches = list(
                    self.source.rglob(f"{split}/run.json")
                )

                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one {split}/run.json "
                        f"in directory; found {len(matches)}"
                    )

                self.locations[split] = matches[0].parent

        else:
            raise FileNotFoundError(self.source)

    def close(self):
        if self.zf is not None:
            self.zf.close()

    def exists(self, split, name):
        if self.zf is not None:
            return (
                self.locations[split] + name
            ) in self.names

        return (
            self.locations[split] / name
        ).is_file()

    def read_bytes(self, split, name):
        if self.zf is not None:
            return self.zf.read(
                self.locations[split] + name
            )

        return (
            self.locations[split] / name
        ).read_bytes()

    def read_json(self, split, name):
        return json.loads(
            self.read_bytes(split, name)
            .decode("utf-8-sig")
        )

    def shards(self, split):
        if self.zf is not None:
            prefix = self.locations[split]

            return sorted(
                n for n in self.names
                if n.startswith(prefix + "shard-")
                and n.endswith(".npz")
                and "/" not in n[len(prefix):]
            )

        return sorted(
            self.locations[split].glob("shard-*.npz")
        )

    def read_shard(self, path):
        if self.zf is not None:
            # Load only one ZIP member into memory.
            stream = io.BytesIO(self.zf.read(path))

            with np.load(
                stream, allow_pickle=False
            ) as data:
                return {
                    k: data[k] for k in data.files
                }

        with np.load(
            path, allow_pickle=False
        ) as data:
            return {
                k: data[k] for k in data.files
            }


def audit_split(
    source, split, coco_root,
    manifest_dir, errors, log
):
    expected_count = COUNTS[split]

    manifest_path = (
        manifest_dir /
        f"{split}_valid_persons.json"
    )

    gt_path = (
        coco_root / "annotations" /
        f"person_keypoints_{split}.json"
    )

    manifest = load_json(manifest_path)
    gt = load_json(gt_path)

    run = source.read_json(split, "run.json")
    status = source.read_json(split, "status.json")
    protocol = run["protocol"]

    ids = [
        int(x) for x in manifest["valid_ann_ids"]
    ]

    expected = {
        int(ann): int(img)
        for img, ann in run["expected_pairs"]
    }

    gt_ann = {
        int(a["id"]): a
        for a in gt["annotations"]
    }

    gt_imgs = {
        int(im["id"]): im
        for im in gt["images"]
    }

    fingerprint = digest_json(run)
    annotation_hash = sha256_file(gt_path)

    def require(condition, message):
        if not condition:
            errors.append(f"{split}: {message}")

    # --------------------------------------------------------
    # 1. Protocol and version checks
    # --------------------------------------------------------

    require(
        run.get("schema_version") == SCHEMA,
        "Wrong cache schema"
    )

    require(
        manifest.get("dataset_split") == split,
        "Wrong manifest split"
    )

    require(
        manifest.get("filter_version")
        == "supervision_filter_v2",
        "Wrong manifest filter"
    )

    require(
        manifest.get("source_annotation") == gt_path.name,
        "Wrong annotation filename"
    )

    require(
        manifest.get("source_annotation_sha256")
        == annotation_hash,
        "Manifest GT SHA256 mismatch"
    )

    require(
        len(ids) == expected_count
        and len(set(ids)) == len(ids),
        "Manifest count or duplicate mismatch"
    )

    require(
        len(expected)
        == len(run["expected_pairs"])
        == expected_count,
        "Expected-pair count or duplicate mismatch"
    )

    require(
        set(expected) == set(ids),
        "Expected IDs differ from approved manifest"
    )

    require(
        len(gt_ann) == len(gt["annotations"]),
        "Duplicate COCO annotation IDs"
    )

    require(
        len(gt_imgs) == len(gt["images"]),
        "Duplicate COCO image IDs"
    )

    required_protocol = {
        "dataset_split": split,
        "scope": "full",
        "data_protocol": "full_train2017_val2017_v1",
        "filter_version": "supervision_filter_v2",
        "bbox_source": "gt",
        "bbox_format": "xyxy",
        "coordinate_system": "original_image",
        "flip_protocol": "single_view",
        "flip_test": False,
        "base_model_eval": True,
        "precision": "float32",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "annotation_sha256": annotation_hash,
        "valid_persons_sha256": digest_json(manifest),
        "full_valid_person_count": expected_count
    }

    for key, value in required_protocol.items():
        require(
            protocol.get(key) == value,
            f"Protocol mismatch: {key}"
        )

    pipeline = protocol.get("pipeline", [])

    require(
        [p.get("type") for p in pipeline]
        == [
            "LoadImage",
            "GetBBoxCenterScale",
            "TopdownAffine",
            "PackPoseInputs"
        ],
        "Unexpected data pipeline"
    )

    # --------------------------------------------------------
    # 2. Cache status and fingerprint
    # --------------------------------------------------------

    require(
        status.get("fingerprint") == fingerprint,
        "Status fingerprint mismatch"
    )

    require(
        status.get("expected") == expected_count,
        "Status expected count mismatch"
    )

    require(
        status.get("count") == expected_count,
        "Status actual count mismatch"
    )

    require(
        status.get("complete") is True,
        "Cache status not complete"
    )

    # --------------------------------------------------------
    # 3. Full GT identity verification
    # --------------------------------------------------------

    for ann_id, image_id in expected.items():
        ann = gt_ann.get(ann_id)

        if (
            ann is None
            or int(ann["image_id"]) != image_id
            or image_id not in gt_imgs
        ):
            errors.append(
                f"{split}: GT identity mismatch "
                f"annotation_id={ann_id}"
            )

            if len(errors) > 100:
                break

    # --------------------------------------------------------
    # 4. Read every NPZ shard and validate its contents
    # --------------------------------------------------------

    seen = set()
    seen_images = set()

    shards = source.shards(split)

    require(
        len(shards) > 0,
        "No NPZ shards found"
    )

    bad_shards = 0

    for index, path in enumerate(shards, 1):

        try:
            batch = source.read_shard(path)

            # Fingerprint belongs to the exact run.json.
            if "fingerprint" not in batch:
                raise ValueError(
                    "Missing shard fingerprint"
                )

            shard_fp = str(
                batch.pop("fingerprint").item()
            )

            if shard_fp != fingerprint:
                raise ValueError(
                    "Shard fingerprint mismatch"
                )

            if set(batch) != set(SHAPES):
                raise ValueError(
                    "Missing or unexpected fields: "
                    f"{sorted(set(batch) ^ set(SHAPES))}"
                )

            n = len(batch["annotation_id"])

            if n == 0:
                raise ValueError("Empty shard")

            # Validate all shapes, dtypes and numeric values.
            for field, shape in SHAPES.items():

                arr = batch[field]

                dtype = (
                    np.int64 if field in INT_FIELDS
                    else np.float32
                )

                if (
                    arr.shape != (n,) + shape
                    or arr.dtype != dtype
                    or not np.isfinite(arr).all()
                ):
                    raise ValueError(
                        f"Invalid shape/dtype/value: {field}"
                    )

            # Confidence + uncertainty must equal 1.
            if not np.allclose(
                batch["confidence"]
                + batch["uncertainty"],
                1.0,
                atol=1e-6
            ):
                raise ValueError(
                    "Confidence/uncertainty mismatch"
                )

            if np.any(
                (batch["uncertainty"] < 0)
                | (batch["uncertainty"] > 1)
            ):
                raise ValueError(
                    "Uncertainty outside [0,1]"
                )

            # Top-1 joint must match maximum uncertainty.
            if not np.array_equal(
                batch["uncertainty"].argmax(axis=1),
                batch["top1_joint"]
            ):
                raise ValueError(
                    "Top-1 joint mismatch"
                )

            if (
                np.any(batch["input_scale"] <= 0)
                or np.any(batch["original_size"] <= 0)
            ):
                raise ValueError(
                    "Invalid coordinate restoration metadata"
                )

            if not np.allclose(
                batch["input_size"],
                [192, 256],
                atol=1e-5
            ):
                raise ValueError(
                    "Unexpected model input size"
                )

            if np.any(
                batch["bbox"][:, 2:]
                <= batch["bbox"][:, :2]
            ):
                raise ValueError(
                    "Invalid xyxy bounding box"
                )

            # Identity and original image size checks.
            for i, (image_id, ann_id) in enumerate(
                zip(
                    batch["image_id"],
                    batch["annotation_id"]
                )
            ):
                image_id = int(image_id)
                ann_id = int(ann_id)

                if ann_id in seen:
                    raise ValueError(
                        f"Duplicate annotation ID {ann_id}"
                    )

                if expected.get(ann_id) != image_id:
                    raise ValueError(
                        f"Wrong image/annotation pair: "
                        f"{image_id}, {ann_id}"
                    )

                image_meta = gt_imgs[image_id]

                if not np.array_equal(
                    batch["original_size"][i],
                    [
                        image_meta["width"],
                        image_meta["height"]
                    ]
                ):
                    raise ValueError(
                        f"Original image size mismatch: "
                        f"{image_id}"
                    )

                seen.add(ann_id)
                seen_images.add(image_id)

        except Exception as exc:
            bad_shards += 1
            errors.append(
                f"{split}: {Path(path).name}: {exc}"
            )

        if (
            index % 500 == 0
            or index == len(shards)
        ):
            log(
                f"{split}: checked "
                f"{index}/{len(shards)} shards, "
                f"unique persons={len(seen)}"
            )

    require(
        len(seen) == expected_count
        and seen == set(expected),
        f"Actual cache incomplete: "
        f"{len(seen)}/{expected_count}"
    )

    # --------------------------------------------------------
    # 5. Verify selected annotations
    # --------------------------------------------------------

    if source.exists(
        split, "selected_annotations.json"
    ):
        selected = source.read_json(
            split, "selected_annotations.json"
        )

        selected_ids = [
            int(a["id"])
            for a in selected["annotations"]
        ]

        require(
            len(selected_ids) == expected_count
            and set(selected_ids) == set(expected),
            "Selected annotations do not match cache"
        )

        require(
            all(
                expected.get(int(a["id"]))
                == int(a["image_id"])
                for a in selected["annotations"]
            ),
            "Selected person/image identity mismatch"
        )

    else:
        errors.append(
            f"{split}: selected_annotations.json missing"
        )

    # --------------------------------------------------------
    # 6. Per-split report
    # --------------------------------------------------------

    summary = {
        "expected_instances": expected_count,
        "verified_instances": len(seen),
        "unique_images": len(seen_images),
        "shard_count": len(shards),
        "invalid_shards": bad_shards,
        "status_complete": status.get("complete"),
        "fingerprint_match": (
            status.get("fingerprint") == fingerprint
        ),
        "manifest_gt_sha256_match": (
            manifest.get("source_annotation_sha256")
            == annotation_hash
        ),
        "protocol": {
            key: protocol.get(key)
            for key in [
                "dataset_split",
                "scope",
                "bbox_source",
                "flip_protocol",
                "flip_test",
                "precision",
                "checkpoint_sha256",
                "config_sha256",
                "git_commit"
            ]
        }
    }

    return summary, seen, seen_images, protocol


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--cache-source",
        type=Path,
        required=True
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
        "--out-dir",
        type=Path,
        default=REPO / "validation_results/step5"
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True, exist_ok=True
    )

    errors = []
    log_lines = []

    report = {
        "task": "Step 5 Full Shared-Cache Validation",
        "checked_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "cache_source": str(args.cache_source),
        "validator_sha256": sha256_file(__file__),
        "overall": "FAIL"
    }

    log_path = args.out_dir / "step5_run.log"
    log_path.write_text("", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_lines.append(str(message))

        with log_path.open(
            "a", encoding="utf-8"
        ) as f:
            f.write(str(message) + "\n")

    source = None

    try:
        source = CacheSource(args.cache_source)

        log(
            f"Cache source: {args.cache_source}"
        )

        summaries = {}
        seen = {}
        image_ids = {}
        protocols = {}

        # Audit both complete splits.
        for split in COUNTS:

            (
                summaries[split],
                seen[split],
                image_ids[split],
                protocols[split]
            ) = audit_split(
                source,
                split,
                args.coco_root,
                args.manifest_dir,
                errors,
                log
            )

        # ----------------------------------------------------
        # Train / Val isolation
        # ----------------------------------------------------

        ann_overlap = len(
            seen["train2017"]
            & seen["val2017"]
        )

        image_overlap = len(
            image_ids["train2017"]
            & image_ids["val2017"]
        )

        if ann_overlap or image_overlap:
            errors.append(
                "Train/Val leakage: "
                f"annotations={ann_overlap}, "
                f"images={image_overlap}"
            )

        # Both splits must use the same model and code.
        shared_protocol_fields = [
            "checkpoint_sha256",
            "config_sha256",
            "implementation_sha256",
            "tracked_code_diff_sha256",
            "filter_version",
            "data_protocol"
        ]

        protocol_consistent = True

        for field in shared_protocol_fields:

            if (
                protocols["train2017"].get(field)
                != protocols["val2017"].get(field)
            ):
                protocol_consistent = False
                errors.append(
                    f"Train/Val protocol mismatch: {field}"
                )

        report.update({
            "splits": summaries,
            "train_val_annotation_overlap": ann_overlap,
            "train_val_image_overlap": image_overlap,
            "protocol_consistent_between_splits": (
                protocol_consistent
            )
        })

    except Exception as exc:
        errors.append(
            f"{type(exc).__name__}: {exc}"
        )

        log(traceback.format_exc())

    finally:
        if source is not None:
            source.close()

        report["error_count"] = len(errors)
        report["errors"] = errors[:100]

        report["overall"] = (
            "PASS" if not errors else "FAIL"
        )

        report_path = (
            args.out_dir / "step5_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        log("\nSTEP 5 FINAL SUMMARY")

        for key in [
            "train_val_annotation_overlap",
            "train_val_image_overlap",
            "protocol_consistent_between_splits",
            "error_count",
            "overall"
        ]:
            if key in report:
                log(
                    f"{key}: {report[key]}"
                )

        if errors:
            log(
                f"First errors: {errors[:5]}"
            )

        log(f"Report: {report_path}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
