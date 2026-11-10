import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = Path(
    r"F:\FYP\data\coco\annotations\person_keypoints_train2017.json"
)
SPLIT = ROOT / "splits/train_dev_split.json"
REPORT = ROOT / "splits/train_dev_data_check_v2.json"
VALID_IDS = ROOT / "splits/train_dev_valid_ann_ids_v2.json"

# 不覆盖已有检查结果
if REPORT.exists() or VALID_IDS.exists():
    raise SystemExit("Output exists. Stop to avoid overwriting.")

original_split_bytes = SPLIT.read_bytes()
split = json.loads(original_split_bytes)
with ANN.open(encoding="utf-8") as f:
    coco = json.load(f)

images = {int(img["id"]): img for img in coco["images"]}
train_ids = split["train_image_ids"]
dev_ids = split["dev_image_ids"]
train = set(train_ids)
dev = set(dev_ids)

assert len(train) == len(train_ids), "Duplicate train image IDs"
assert len(dev) == len(dev_ids), "Duplicate dev image IDs"
assert not train & dev, "Train/dev image overlap"
assert train | dev == set(images), "Image coverage mismatch"

def finite_numbers(values):
    return all(
        isinstance(v, (int, float)) and math.isfinite(v)
        for v in values
    )

def invalid_reasons(ann, img):
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
        # 对齐仓库现有读取代码的附加检查
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

        # 与仓库一致，将边界裁剪到 [0, width/height - 1]
        clip = lambda value, upper: max(0, min(value, upper))
        x1 = clip(x, img["width"] - 1)
        y1 = clip(y, img["height"] - 1)
        x2 = clip(x + w, img["width"] - 1)
        y2 = clip(y + h, img["height"] - 1)
        if x2 <= x1 or y2 <= y1:
            reasons.append("invalid_clipped_bbox")

    return reasons

groups = {
    "train": sorted(train),
    "dev": sorted(dev),
}
stats = {}
valid_ann_ids = {"train": [], "dev": []}

for name, ids in groups.items():
    stats[name] = {
        "image_count": len(ids),
        "total_person_annotations": 0,
        "valid_persons": 0,
        "filtered_persons": 0,
        "filter_reason_counts": Counter(),
        "labeled_keypoints": 0,
        "occluded_labeled_keypoints": 0,
    }

persons_per_image = {image_id: 0 for image_id in images}
seen_ann_ids = set()

for ann in coco["annotations"]:
    ann_id = int(ann["id"])
    assert ann_id not in seen_ann_ids, "Duplicate annotation ID"
    seen_ann_ids.add(ann_id)

    image_id = int(ann["image_id"])
    assert image_id in images, "Unknown image ID"
    name = "train" if image_id in train else "dev"
    s = stats[name]
    s["total_person_annotations"] += 1

    reasons = invalid_reasons(ann, images[image_id])
    if reasons:
        s["filtered_persons"] += 1
        s["filter_reason_counts"].update(reasons)
        continue

    valid_ann_ids[name].append(ann_id)
    s["valid_persons"] += 1
    persons_per_image[image_id] += 1

    visibility = ann["keypoints"][2::3]
    s["labeled_keypoints"] += sum(v > 0 for v in visibility)
    s["occluded_labeled_keypoints"] += sum(v == 1 for v in visibility)

for name, ids in groups.items():
    s = stats[name]
    histogram = Counter(persons_per_image[i] for i in ids)

    s["images_without_valid_persons"] = histogram[0]
    s["mean_valid_persons_per_image"] = s["valid_persons"] / len(ids)
    s["occluded_fraction_among_labeled_keypoints"] = (
        s["occluded_labeled_keypoints"] / s["labeled_keypoints"]
        if s["labeled_keypoints"] else None
    )
    s["valid_persons_per_image_histogram"] = {
        str(k): histogram[k] for k in sorted(histogram)
    }
    s["filter_reason_counts"] = dict(s["filter_reason_counts"])
    valid_ann_ids[name].sort()

    assert (
        s["valid_persons"] + s["filtered_persons"]
        == s["total_person_annotations"]
    )

# 验证脚本没有修改原 split
assert SPLIT.read_bytes() == original_split_bytes
split_hash = hashlib.sha256(original_split_bytes).hexdigest()

report = {
    "source_annotation": ANN.name,
    "split_file": SPLIT.name,
    "split_sha256": split_hash,
    "image_id_overlap": len(train & dev),
    "full_image_coverage": train | dev == set(images),
    "split_unchanged": True,
    "notes": [
        "Statistics apply only to train2017 supervision.",
        "No val2017 annotations are read or modified.",
        "Occlusion fraction is v=1 divided by v>0, among valid persons.",
        "Filter reasons can overlap; do not sum them as filtered persons.",
        "Future data builders must use the valid annotation IDs.",
    ],
    "statistics": stats,
}

manifest = {
    "source_annotation": ANN.name,
    "split_sha256": split_hash,
    "filter_version": "supervision_filter_v2",
    "train_valid_ann_ids": valid_ann_ids["train"],
    "dev_valid_ann_ids": valid_ann_ids["dev"],
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
with VALID_IDS.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
print("\nSaved:", REPORT)
print("Saved:", VALID_IDS)
print("\nPASS: existing image split unchanged.")