import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--ann", required=True, type=Path)
    args = parser.parse_args()

    raw = args.detections.read_bytes()
    detections = json.loads(raw)

    with args.ann.open(encoding="utf-8-sig") as f:
        coco = json.load(f)

    if not isinstance(detections, list):
        raise ValueError("Detection JSON must be a list")

    images = {img["id"]: img for img in coco["images"]}
    categories = Counter()
    invalid_reasons = Counter()
    eligible = []
    seen = set()
    duplicate_records = 0

    for index, det in enumerate(detections):
        category = det.get("category_id")
        categories[str(category)] += 1
        reasons = []

        if category != 1:
            reasons.append("not_person")

        image_id = det.get("image_id")
        if image_id not in images:
            reasons.append("unknown_image_id")

        bbox = det.get("bbox")
        bbox_ok = (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(v, (int, float)) and math.isfinite(v)
                for v in bbox
            )
        )
        if not bbox_ok:
            reasons.append("invalid_bbox_format_or_values")
        elif bbox[2] <= 0 or bbox[3] <= 0:
            reasons.append("nonpositive_bbox_width_or_height")

        score = det.get("score")
        score_ok = (
            isinstance(score, (int, float))
            and math.isfinite(score)
            and 0 <= score <= 1
        )
        if not score_ok:
            reasons.append("invalid_score")

        if reasons:
            invalid_reasons.update(reasons)
            continue

        # Count exact repeated records, but do not delete any.
        key = (image_id, category, *bbox, score)
        if key in seen:
            duplicate_records += 1
        seen.add(key)
        eligible.append(det)

    threshold_stats = {}
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.5]:
        selected = [d for d in eligible if d["score"] >= threshold]
        threshold_stats[str(threshold)] = {
            "records": len(selected),
            "images_with_detections": len(
                {d["image_id"] for d in selected}
            ),
        }

    report = {
        "status": "inspection_only_not_final_protocol",
        "detection_file": args.detections.name,
        "detection_file_sha256": hashlib.sha256(raw).hexdigest(),
        "annotation_file": args.ann.name,
        "total_val_images": len(images),
        "raw_detection_records": len(detections),
        "category_counts": dict(categories),
        "structurally_valid_person_records": len(eligible),
        "invalid_records": len(detections) - len(eligible),
        "invalid_reason_counts_nonexclusive": dict(invalid_reasons),
        "exact_duplicate_records_among_valid": duplicate_records,
        "candidate_threshold_statistics": threshold_stats,
        "notes": [
            "Threshold statistics use score >= threshold.",
            "No source records were modified or removed.",
            "No additional NMS was performed.",
            "Duplicate counts are informational; no deduplication applied.",
            "GT image metadata was used only to check image membership.",
            "No GT person list was used to filter detections.",
            "Final threshold and filtering/NMS require team confirmation.",
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()