import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = Path(
    r"F:\FYP\data\coco\annotations\person_keypoints_train2017.json"
)
OUT_DIR = ROOT / "splits"
SPLIT_PATH = OUT_DIR / "train_dev_split.json"
REPORT_PATH = OUT_DIR / "train_dev_split_report.json"

SEED = 42
TRAIN_RATIO = 0.9

# 防止覆盖已有正式划分
if SPLIT_PATH.exists() or REPORT_PATH.exists():
    raise SystemExit(
        "Output already exists. Stop to avoid overwriting an existing split."
    )

with ANN_PATH.open(encoding="utf-8") as f:
    coco = json.load(f)

# 排序后再使用固定种子打乱，保证相同输入可重复生成
image_ids = sorted(int(img["id"]) for img in coco["images"])
assert len(image_ids) == len(set(image_ids)), "Duplicate image IDs"

shuffled = image_ids.copy()
random.Random(SEED).shuffle(shuffled)

n_train = int(len(shuffled) * TRAIN_RATIO)
train_ids = sorted(shuffled[:n_train])
dev_ids = sorted(shuffled[n_train:])

train_set = set(train_ids)
dev_set = set(dev_ids)
all_ids = set(image_ids)

assert not train_set & dev_set, "Train/dev overlap"
assert train_set | dev_set == all_ids, "Missing images"

# 所有人体按所属图片进入集合。
# 下面的过滤只决定有效人体统计，不改变图片划分。
total = Counter()
valid = Counter()
filtered = Counter()

for ann in coco["annotations"]:
    image_id = int(ann["image_id"])
    assert image_id in all_ids, "Annotation refers to unknown image"
    subset = "train" if image_id in train_set else "dev"
    total[subset] += 1

    reason = None
    if ann.get("category_id") != 1:
        reason = "not_person"
    elif ann.get("iscrowd", 0):
        reason = "crowd"
    else:
        kpts = ann.get("keypoints", [])
        area = float(ann.get("area", 0))

        if len(kpts) != 51:
            reason = "invalid_keypoint_length"
        elif not any(v > 0 for v in kpts[2::3]):
            reason = "no_labeled_keypoints"
        elif not math.isfinite(area) or area <= 0:
            reason = "invalid_area"

    if reason is None:
        valid[subset] += 1
    else:
        filtered[f"{subset}:{reason}"] += 1

split = {
    "version": "coco_train2017_image_split_v1",
    "source_annotation": ANN_PATH.name,
    "seed": SEED,
    "train_ratio": TRAIN_RATIO,
    "split_unit": "image_id",
    "image_scope": "all images listed in the source annotation",
    "train_image_ids": train_ids,
    "dev_image_ids": dev_ids,
}

report = {
    "total_images": len(image_ids),
    "train_images": len(train_ids),
    "dev_images": len(dev_ids),
    "train_person_annotations": total["train"],
    "dev_person_annotations": total["dev"],
    "train_valid_persons": valid["train"],
    "dev_valid_persons": valid["dev"],
    "valid_person_rule": (
        "category_id=1, iscrowd=0, 17 keypoints, "
        "at least one v>0, finite positive GT area"
    ),
    "filter_counts_exclusive": dict(sorted(filtered.items())),
    "image_id_overlap": len(train_set & dev_set),
    "full_image_coverage": train_set | dev_set == all_ids,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

with SPLIT_PATH.open("w", encoding="utf-8") as f:
    json.dump(split, f, ensure_ascii=False, indent=2)

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
print("\nSaved:", SPLIT_PATH)
print("Saved:", REPORT_PATH)
print("\nPASS: no overlap; all source images covered.")