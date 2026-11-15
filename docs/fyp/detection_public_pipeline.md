# Detection-box public pipeline handoff

## Scope and lineage

Integration branch: feature/detection-public-pipeline.
- L input adapter/tests: lin, 7092688 (imported unchanged).
- T protocol: feature/intervention-interface, c720e00; head API remains 6fd5b3c.
- M CPU evaluator: mai/eval-and-module-c, 9b76955, with integration corrections below.
- N GT cache baseline: feature/shared-cache, 345a814.

Only public infrastructure is implemented here; A/B/C decision models are not trained.
Full test-dev inference is not run before model selection.

## Tasks

1: evaluation.md and interfaces.md fix split roles, source artifact, scoring and provenance limits.
2A: scripts/read_val_detection_boxes.py retains every approved record and stable string ID.
2B: build_detection_cache.DetectionDataset maps L's records into the deterministic model pipeline;
    existing frozen head APIs are reused, with no intervention math copied or changed.
2C: detection_cache.py adds a separate detection schema; legacy shared_cache.py stays unchanged.
2D/3: real sample forward checks, check_detection_outputs.py and CPU/reference cross-checks.
4A: build_detection_cache.py writes reference and baseline predictions in one shared-backbone pass.
4B: evaluate_detection_cache.py exports all modes and verifies reference vs CPU results.
5: audit_shared_caches.py validates full manifests, identities, boxes/scores and image disjointness.

## Cache fields and identity

DetectionCache uses schema cigpose-detection-cache-v1, not the legacy GT schema.
NPZ fields (float32 unless stated):
- features [B,17,256], uncertainty/confidence [B,17]
- P_0_coords [B,17,2], P_0_scores [B,17]
- bbox [B,4] original-image XYXY, bbox_score [B]
- input_center/input_scale/input_size/original_size [B,2]
- image_id/source_detection_index/top1_joint/original_joint [B] int64
- original_coords/top1_coords [B,17,2], corresponding *_scores [B,17]

Baseline predictions are additional model outputs for reproducible evaluation, not GT labels.
No GT annotation ID, GT keypoints, GT area, true OKS or true gain is in the cache.
Source index is never an annotation ID. Full string IDs are in detection_manifest.json;
cache.instance_id(index) reconstructs the exact same string from source SHA plus index.

```python
import sys
sys.path[:0] = ['mmpose/analysis/fyp', 'scripts']
from detection_cache import ReferenceCache
cache = ReferenceCache('/outputs/val2017-detection')
for batch in cache.iter_shards():
    ids = [cache.instance_id(i) for i in batch['source_detection_index']]
    features = batch['features']
```

Cache creation fingerprints resolved configuration, weights, detection/annotation hashes,
code hashes, batch size and ID manifest. Shards are atomic NPZ files with allow_pickle=False;
writer lock prevents concurrent writes. --resume rejects changed protocols. A crash may
leave a stale lock: verify that the writer process has exited before manually removing it.
Do not modify code in the isolated run snapshot while resuming a run.

## CPU integration changes relative to M's branch

- Explicit detection protocol label; retain supplied bbox scores.
- Require transformed scales rather than silently falling back to raw-box area.
- Match official pairwise area average with epsilon (no area clamp to 1).
- Match float32 sigma, operation order and output precision of the repository OKS NMS.
- Match reference keypoint-score accumulation/order, retaining score precision for sorting.
- Reject missing/duplicate per-instance record IDs, including duplicate IDs across batches.
- Use the reference float32 sigmas in COCOeval and preserve all 5,000 evaluation images.
- Retain the original prediction bbox in final JSON, matching COCO's handling of
  unmatched detection area ranges; do not replace it with a keypoint-derived area.

Regression tests include resume, protocol mismatch, duplicate IDs, wrong-image identity,
GT-field rejection, lock/roundtrip and unequal/small-area NMS.

## Reproducible checks

```bash
python tests/test_val_detection_reader.py
python mmpose/analysis/fyp/tests/test_detection_pipeline.py
python mmpose/analysis/fyp/check_detection_outputs.py --cache SMOKE_CACHE \
  --ann COCO/annotations/person_keypoints_val2017.json --images COCO/val2017 --output VISUAL_CHECKS
python mmpose/analysis/fyp/audit_shared_caches.py --train-cache TRAIN_GT_CACHE \
  --val-cache VAL_DETECTION_CACHE --data COCO --output audit.json
```

Use evaluation.md for full generation/evaluation commands. Sample OKS matching is
independent best-box IoU >=0.5 for diagnostics only: it is not a validated full-val
supervision matcher, and it never removes detections from AP/AR evaluation.

## Final-test prerequisites

The current local inventory did not locate test2017 images, image_info_test-dev2017.json
or test-dev detections in the inspected WSL projects, D:/15790/FYP, desktop and downloads.
This is a search of specified locations, not proof about every disk/path.
Do not download or submit test data as part of val infrastructure acceptance.
The fixed val detection file does not establish upstream detector config/checkpoint.
Acquire matching final-test detector provenance/artifacts before claiming a matched
val/test-dev detector protocol. A change of detector requires matched val reruns.
