# Main evaluation protocol: detection boxes / single-view

Updated 2026-11-16. This replaces the former GT-box **main evaluation** protocol.

## Dataset roles

- train2017: full approved GT-person manifest, GT crops, offline gain supervision.
- val2017: fixed detection input, tuning/model selection/ablation. Results are development-informed validation results.
- test-dev2017: final submission after settings are locked. No test-driven tuning.
- Frozen CIGPose-m checkpoint; float32; eval/inference mode; 256x192 HxW; COCO 17 points.
- flip_test=False everywhere, no flipped branch or prediction fusion.
- GT-box validation results remain separately labelled diagnostics, not the main comparison.

## Fixed val detection input

File: COCO_val2017_detections_AP_H_56_person.json
SHA-256: 53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59
104,125 records / 3,893 images, all retained. No additional score filtering, box NMS,
deduplication, clipping or GT-manifest filtering. Source bbox is original-image XYWH.
L's explicit bbox_xyxy is converted to float32 for the model; source values remain
available in the fixed input file. Detection score is likewise represented in float32
for model evaluation, without changing its meaning or thresholding it.

ID: val2017:det:<full SHA-256>:<zero-based source index padded to six digits>.
The numeric source index used by CocoMetric is a detection index, NOT annotation_id.

The supplied detection artifact is fixed, but its upstream detector config and weights
are **not established**. Do not claim that the test-dev detector protocol is already
reproduced. Obtain corresponding provenance/detection artifacts before final test;
if a different detector is chosen, rerun matched val comparisons with that detector.

## Canonical execution

Config: mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_coco-256x192_detection-single-view.py
Canonical input adapter: scripts/read_val_detection_boxes.py + build_detection_cache.py.
The adapter consumes L's explicit XYXY view without GT data in model inputs.
The original training configuration and GT diagnostic config are retained.

```bash
python mmpose/analysis/fyp/build_detection_cache.py \
  --detections /data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json \
  --ann /data/coco/annotations/person_keypoints_val2017.json \
  --images /data/coco/val2017 \
  --checkpoint /checkpoints/cigpose-m_coco_256x192.pth \
  --output /outputs/val2017-detection --batch-size 32 --workers 2
```

First use --smoke and a separate directory for L's six-instance integration set.
Use --resume only with identical fingerprint/settings. A stale writer lock may only
be removed after confirming no process is writing. Never edit run.json to bypass checks.

## Scoring and post-processing

- bbox_keypoint: detector score multiplied by mean decoded keypoint score strictly >0.2;
  zero if no keypoint exceeds the threshold.
- OKS NMS threshold 0.9, once only.
- NMS area is the product of transformed input_scale / bbox_scales, not GT area.
- Full original val2017 annotation file; evaluate all 5,000 images, including the
  1,107 without detections. Do not restrict evaluation to detected or matched images.
- Original CIGPose, P0, Top-1 alpha=1 and all ABC variants share these settings.
- Do not claim exact Original/Top-1 equivalence without the numerical report.

## CPU/reference cross-check and baseline export

```bash
python mmpose/analysis/fyp/evaluate_detection_cache.py \
  --cache /outputs/val2017-detection \
  --ann /data/coco/annotations/person_keypoints_val2017.json \
  --output /outputs/detection-baselines
```

This uses the installed MMPose CocoMetric as reference and independently checks M's
CPU pre-NMS and final-JSON paths: all ten metrics and postprocessed records. No GPU
forward is repeated. The three raw JSON files retain detection IDs and source indices.

M can independently evaluate delivered final JSON (no MMCV required):

```bash
python mmpose/analysis/fyp/eval_unified.py --ann-file person_keypoints_val2017.json \
  --pred-file predictions.keypoints.json --stage coco_json --exp-name original_detection
```

pre_nms requires input_scale/bbox_scales, detector score and unique instance identity.
Missing scales must fail, never silently substitute a different area. Final COCO JSON
has already undergone scoring and NMS and must not be postprocessed again.
The CPU entry currently requires numpy, torch and pycocotools; CUDA/MMCV are not required.

## Supervision boundary

Instance OKS/gain uses corresponding GT area/visibility and L's float64 COCO sigmas.
This is distinct from pose NMS, which compares predicted instances using input-scale area.
GT matching in check_detection_outputs.py is a sample-only diagnostic, not a production
training matcher. It never changes formal AP/AR inputs. No GT keypoints, true gain or
GT matching result is passed into ABC model inputs.
