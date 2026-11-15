"""Fixed val2017 detection input. Standard library only; no score filtering/NMS.

Public entry point:
    metadata, instances = load_detection_instances(detections, annotation, image_root)

bbox is original-image XYWH; bbox_xyxy is an explicitly named derived view.
Neither representation is clipped or padded. The model pipeline owns cropping.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

APPROVED_SHA256 = "53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59"
APPROVED_RECORD_COUNT = 104125
SCHEMA_VERSION = "val2017_fixed_detection_v1"


def detection_instance_id(source_sha256, source_index):
    """Original 0-based source index, not filtered rank or DataLoader position."""
    return f"val2017:det:{source_sha256}:{source_index:06d}"


def _load(detection_path, annotation_path, image_root, expected_sha256, expected_count):
    detection_path, annotation_path = Path(detection_path), Path(annotation_path)
    raw = detection_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Detection source SHA-256 differs from the fixed protocol")
    detections = json.loads(raw)
    if not isinstance(detections, list) or len(detections) != expected_count:
        raise ValueError("Unexpected detection structure or count")
    annotation_bytes = annotation_path.read_bytes()
    coco = json.loads(annotation_bytes)
    # Only image metadata is read. GT person IDs/keypoints/areas do not filter boxes.
    images = {}
    for image in coco["images"]:
        image_id = image["id"]
        if type(image_id) is not int or image_id in images:
            raise ValueError("Invalid or duplicate image ID")
        if not isinstance(image.get("file_name"), str):
            raise ValueError("Image file_name is required")
        images[image_id] = image

    instances = []
    for index, det in enumerate(detections):
        image_id = det.get("image_id")
        if type(image_id) is not int or image_id not in images:
            raise ValueError(f"Unknown/invalid image_id at source index {index}")
        if det.get("category_id") != 1:
            raise ValueError(f"Non-person record at source index {index}; not silently removed")
        bbox = det.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                not all(type(v) in (int, float) and math.isfinite(v) for v in bbox)):
            raise ValueError(f"Invalid bbox at source index {index}")
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Nonpositive bbox size at source index {index}")
        score = det.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score at source index {index}")
        image = images[image_id]
        iid = detection_instance_id(digest, index)
        record = {
            "dataset_split": "val2017", "image_id": image_id,
            "detection_instance_id": iid, "instance_id": iid,
            "source_detection_index": index, "category_id": 1,
            "bbox": list(bbox), "bbox_format": "xywh",
            "bbox_xyxy": [x, y, x + w, y + h],
            "bbox_score": score,
            "image_file_name": image["file_name"],
            "image_width": image["width"], "image_height": image["height"],
        }
        if image_root is not None:
            record["image_path"] = str(Path(image_root) / image["file_name"])
        instances.append(record)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_split": "val2017", "box_protocol": "detection",
        "source_detection_file": detection_path.name,
        "source_detection_sha256": digest,
        "source_annotation": annotation_path.name,
        "source_annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "bbox_source": "fixed_precomputed_COCO_val2017_detections_AP_H_56_person",
        "source_record_count": len(detections), "selected_record_count": len(instances),
        "score_threshold": 0.0, "additional_score_filtering": False,
        "additional_bbox_nms": False, "additional_deduplication": False,
        "bbox_format": "xywh", "bbox_coordinates": "original_image",
        "bbox_clipping_applied": False,
        "id_rule": "val2017:det:<full raw-file sha256>:<zero-based source index, width 6>",
        "upstream_detector_config": None, "upstream_detector_checkpoint": None,
        "upstream_nms": "not inferred; preserve the supplied fixed detector output",
        "pose_scoring_and_pose_nms": "handled separately by the shared evaluator",
        "old_90_10_split_used": False, "gt_person_list_used_for_filtering": False,
    }
    return metadata, instances


def load_detection_instances(detection_path, annotation_path, image_root=None):
    """Read the approved file without dropping/reordering/modifying any detections.

    image_root, when supplied, is the directory containing val2017 JPG files.
    GT annotation_id is intentionally absent: detections are not GT instances.
    """
    if Path(annotation_path).name != "person_keypoints_val2017.json":
        raise ValueError("Use person_keypoints_val2017.json for val2017 image metadata")
    return _load(detection_path, annotation_path, image_root,
                 APPROVED_SHA256, APPROVED_RECORD_COUNT)


def select_samples(instances, sample_images=3, boxes_per_image=2):
    """Sample export only; does not change the full inference input list."""
    chosen_images, samples = {}, []
    for record in instances:
        image_id = record["image_id"]
        if image_id not in chosen_images:
            if len(chosen_images) >= sample_images:
                continue
            chosen_images[image_id] = 0
        if chosen_images[image_id] < boxes_per_image:
            samples.append(record.copy())
            chosen_images[image_id] += 1
    return samples


def export_delivery(metadata, instances, out_dir, image_root=None,
                    sample_images=3, boxes_per_image=2):
    out_dir = Path(out_dir)
    paths = [out_dir / name for name in (
        "val2017_detection_manifest.json", "val2017_detection_report.json",
        "val2017_detection_samples.json")]
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError("Choose a new --out-dir; outputs exist: " + ", ".join(existing))
    ids = [r["detection_instance_id"] for r in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate detection instance IDs")
    samples = select_samples(instances, sample_images, boxes_per_image)
    if image_root is not None:
        missing = [r["image_path"] for r in samples if not Path(r["image_path"]).is_file()]
        if missing:
            raise FileNotFoundError("Sample images are missing: " + ", ".join(missing))
    histogram = Counter(r["image_id"] for r in instances)
    report = {
        **metadata,
        "images_with_detections": len(histogram),
        "unique_instance_ids": len(set(ids)),
        "sample_images": len({r["image_id"] for r in samples}),
        "sample_instances": len(samples),
        "sample_image_files_checked": image_root is not None,
        "all_image_files_checked": False,
        "real_model_forward_checked": False,
        "checks": {"all_source_records_retained": len(instances) == metadata["source_record_count"],
                   "original_source_order_preserved": all(r["source_detection_index"] == i
                                                         for i, r in enumerate(instances)),
                   "unique_instance_ids": len(ids) == len(set(ids))},
    }
    manifest = {**metadata, "detection_instance_ids": ids}
    sample_payload = {
        "metadata": metadata,
        "sample_only": True,
        "sampling_rule": "First distinct images in source order; first boxes_per_image records per image",
        "sample_image_files_checked": image_root is not None,
        "instances": samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, value in zip(paths, [manifest, report, sample_payload]):
        with path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    return report, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, help="Directory containing val2017 JPGs")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-images", type=int, default=3)
    parser.add_argument("--boxes-per-image", type=int, default=2)
    args = parser.parse_args()
    if args.sample_images <= 0 or args.boxes_per_image <= 0:
        parser.error("Sample sizes must be positive")
    meta, records = load_detection_instances(args.detections, args.ann, args.image_root)
    report, paths = export_delivery(meta, records, args.out_dir, args.image_root,
                                    args.sample_images, args.boxes_per_image)
    print("Source records:", meta["source_record_count"])
    print("Retained records:", meta["selected_record_count"])
    print("Images with detections:", report["images_with_detections"])
    print("Unique instance IDs:", report["unique_instance_ids"])
    print("Sample instances:", report["sample_instances"])
    print("Sample image files checked:", report["sample_image_files_checked"])
    print("Score filtering: none; additional bbox NMS: none; bbox clipping: none")
    for p in paths:
        print("Saved:", p)
    print("PASS: fixed detection input exported; real model forward still pending.")


if __name__ == "__main__":
    main()
