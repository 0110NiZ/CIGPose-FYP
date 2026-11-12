"""Shared portable JSON identity and supervision filtering. Standard library only."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SPLIT_HASH_METHOD = "sha256-json-sort-keys-compact-utf8-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def split_sha256(split):
    """Hash parsed JSON, ignoring formatting, line endings and object-key order.

    Array order, actual image IDs and metadata remain part of the identity.
    """
    canonical = json.dumps(split, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_metadata(split):
    return {"split_sha256": split_sha256(split),
            "split_hash_method": SPLIT_HASH_METHOD}


def verify_manifest(split, manifest):
    if manifest.get("split_hash_method") != SPLIT_HASH_METHOD:
        raise ValueError("Legacy/unknown checksum method: migrate or regenerate the manifest first.")
    if manifest.get("split_sha256") != split_sha256(split):
        raise ValueError("Manifest does not match the parsed split content")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def check_outputs(paths, overwrite=False):
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; choose another --out-dir or use --overwrite for reports: "
                              + ", ".join(existing))


def finite_numbers(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def invalid_reasons(ann, img):
    # Same filtering rules as the reviewed v2 script; unchanged by this revision.
    reasons = []
    if ann.get("category_id") != 1:
        reasons.append("not_person")
    if ann.get("iscrowd", 0):
        reasons.append("iscrowd")
    kpts = ann.get("keypoints", [])
    if len(kpts) != 51 or not finite_numbers(kpts):
        reasons.append("invalid_keypoints")
    else:
        if not any(v > 0 for v in kpts[2::3]):
            reasons.append("no_labeled_keypoints")
        if ann.get("num_keypoints") == 0:
            reasons.append("num_keypoints_zero")
        if max(kpts[0::3] + kpts[1::3]) <= 0:
            reasons.append("nonpositive_keypoint_coordinates")
    area = ann.get("area", 0)
    if not finite_numbers([area]) or area <= 0:
        reasons.append("invalid_area")
    bbox = ann.get("bbox", [])
    if len(bbox) != 4 or not finite_numbers(bbox):
        reasons.append("invalid_bbox")
    else:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            reasons.append("invalid_bbox")
        clip = lambda v, upper: max(0, min(v, upper))
        x1, y1 = clip(x, img["width"] - 1), clip(y, img["height"] - 1)
        x2, y2 = clip(x + w, img["width"] - 1), clip(y + h, img["height"] - 1)
        if x2 <= x1 or y2 <= y1:
            reasons.append("invalid_clipped_bbox")
    return reasons


def audit_coco(coco, split):
    images = {int(img["id"]): img for img in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate source image IDs")
    groups = {name: split[name + "_image_ids"] for name in ("train", "dev")}
    sets = {name: set(ids) for name, ids in groups.items()}
    for name, ids in groups.items():
        if len(sets[name]) != len(ids) or not ids:
            raise ValueError("Duplicate IDs or empty subset: " + name)
    if sets["train"] & sets["dev"]:
        raise ValueError("Train/dev overlap")
    if sets["train"] | sets["dev"] != set(images):
        raise ValueError("Image coverage mismatch")
    stats = {name: {"image_count": len(ids), "total_person_annotations": 0,
                   "valid_persons": 0, "filtered_persons": 0,
                   "filter_reason_counts": Counter(), "labeled_keypoints": 0,
                   "occluded_labeled_keypoints": 0} for name, ids in groups.items()}
    valid = {"train": [], "dev": []}
    persons = Counter()
    seen = set()
    for ann in coco["annotations"]:
        ann_id, image_id = int(ann["id"]), int(ann["image_id"])
        if ann_id in seen or image_id not in images:
            raise ValueError("Duplicate annotation ID or unknown image")
        seen.add(ann_id)
        name = "train" if image_id in sets["train"] else "dev"
        s = stats[name]
        s["total_person_annotations"] += 1
        reasons = invalid_reasons(ann, images[image_id])
        if reasons:
            s["filtered_persons"] += 1
            s["filter_reason_counts"].update(reasons)
            continue
        valid[name].append(ann_id)
        s["valid_persons"] += 1
        persons[image_id] += 1
        vis = ann["keypoints"][2::3]
        s["labeled_keypoints"] += sum(v > 0 for v in vis)
        s["occluded_labeled_keypoints"] += sum(v == 1 for v in vis)
    for name, ids in groups.items():
        s = stats[name]
        hist = Counter(persons[i] for i in ids)
        s.update(images_without_valid_persons=hist[0],
                 mean_valid_persons_per_image=s["valid_persons"] / len(ids),
                 occluded_fraction_among_labeled_keypoints=(
                     s["occluded_labeled_keypoints"] / s["labeled_keypoints"]
                     if s["labeled_keypoints"] else None),
                 valid_persons_per_image_histogram={str(k): hist[k] for k in sorted(hist)})
        s["filter_reason_counts"] = dict(s["filter_reason_counts"])
        valid[name].sort()
    report = {"source_annotation": split["source_annotation"], **hash_metadata(split),
              "image_id_overlap": 0, "full_image_coverage": True,
              "split_unchanged": True,
              "notes": ["Statistics apply only to train2017 supervision.",
                        "No val2017 annotations are read or modified.",
                        "Occlusion fraction is v=1 divided by v>0, among valid persons.",
                        "Filter reasons overlap; do not sum them as filtered persons.",
                        "Data builders must use the valid annotation IDs.",
                        "Split checksum is based on canonical parsed JSON, not file bytes."],
              "statistics": stats}
    manifest = {"source_annotation": split["source_annotation"], **hash_metadata(split),
                "filter_version": "supervision_filter_v2",
                "train_valid_ann_ids": valid["train"], "dev_valid_ann_ids": valid["dev"]}
    return report, manifest
