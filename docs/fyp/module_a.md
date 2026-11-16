# Module A: N-side implementation and Y label handoff

Base: `public-tasks` commit `66edb12`. Development branch: `feature/module-a`.

Current editable training code lives in [`Module_A/`](../../Module_A/README.md).
The old `mmpose/analysis/fyp/train_a.py` and `gain_predictor.py` have been removed.
Public data checks, label tools and evaluation helpers remain dependencies; do not
delete them. The layout cleanup did not train a model or change caches/labels.

## Scope and status

Update: the user has now completed the full seed-42, 20-epoch run. The best
validation checkpoint is epoch 10, AP 73.5190 and AR 78.9814 on the 0–100 scale.
See [run results and controlled comparison](../../experiments/module_a/seed42_v2/README.md).
The label acceptance and layout checks described below were earlier, non-training stages.

N implements the input reader/encoder, MLP, training entry, and detection-val export.
Y implements candidate generation, gain-label generation, and independent label audit.
Y's original labels were audited, then N regenerated and accepted the full V2
labels under the approved deterministic precision profile. That acceptance did not
perform optimization; the subsequent user-run training is reported separately. No
AP improvement is inferred from interface tests. Synthetic unit-test labels are
temporary fixtures only. The real-cache interface check uses an untrained network,
does not save its weights, and does not report AP/AR.

Shared forward/cache entries now enforce the [V2 precision policy](precision_policy.md);
the base head, weights and evaluation rules remain unchanged. Training uses all
149,813 approved train GT persons. Validation uses all 104,125 approved detections,
with AP/AR over the original 5,000 val images. Base CIGPose remains frozen/eval,
single-view, float32; A-only selects one joint, alpha=1, always accepts. Negative
predicted gains do not trigger fallback to P0 (that would introduce C).

## Files owned by N

The daily-use files are under `Module_A/`; shared dependencies stay under
`mmpose/analysis/fyp/`:

| File | Responsibility |
|---|---|
| Module_A/model.py | Actual MLP definition, class MLP; also used by shared A helpers |
| Module_A/config.py | Default data paths and training settings |
| Module_A/data.py | Reuses the audited a_dataset reader |
| Module_A/train.py | Training loop, preflight by default, explicit opt-in for optimization |
| Module_A/evaluate.py | Standalone full-val evaluation using this directory's MLP |
| Module_A/common.py | Locates the shared project code |
| mmpose/analysis/fyp/a_contract.py | Label format, provenance/identity checks and input whitelist |
| mmpose/analysis/fyp/a_dataset.py | Memory-bounded reading, ID join and metadata template |
| mmpose/analysis/fyp/a_runtime.py | Frozen base, compatibility and checkpoint helpers |
| mmpose/analysis/fyp/predict_a.py | Shared candidate export and unified evaluation functions |
| mmpose/analysis/fyp/check_a_interface.py | Read-only cache and forward checks |
| mmpose/analysis/fyp/tests/test_module_a.py | Contract, identity, model and gradient tests |

`build_a_labels_legacy.py` generates the independent V2 label version;
`verify_a_labels_legacy.py` audits it before marking metadata complete.
See [current status and training plan](project_status_20261116.md) and
`experiments/module_a/v2/` for accepted results and remaining numerical boundaries.

## Label interface and accepted V2 delivery

The array schema remains compatible with the original handoff. The accepted V2
delivery additionally records the precision profile and verified source snapshots.
A concise Chinese handoff is in
`module_a_handoff_zh.md`.

Y reads the same full train reference cache. For every annotation, create 17
independent candidates from its untouched `features`, alpha=1, COCO joint order
0..16. Reuse `head.cim.apply_intervention`, `head.predict_from_joint_features`,
the public decoding/restoration path and `oks_utility.compute_gain`. P0 and every
candidate must be compared with the **same** GT annotation in original coordinates.
Use GT annotation area and visibility exactly as the shared OKS implementation does.

Delivery folder:

```text
train_a_labels/
  metadata.json
  shard-00000.npz
  shard-00001.npz
  ...
  validation_report.json   # Y's independent audit; not an input feature
```

Each NPZ has exactly these five arrays, no pickle:

| Field | Shape | Dtype |
|---|---|---|
| image_id | [N] | int64 |
| annotation_id | [N] | int64 |
| oks_before | [N] | float32 or float64 |
| oks_after | [N,17] | float32 or float64 |
| gain | [N,17] | float32 or float64 |

`gain = oks_after - oks_before[:, None]`, absolute tolerance 2e-6. Preserve signs.
The shard count/size/order is independent of the reference-cache shard layout.
N joins by annotation ID inside the explicitly checked train2017 split and also
checks image ID. No row-number join. All approved persons must appear exactly once.

Use `a_dataset.py --write-contract` to create a metadata template with the **actual**
train cache fingerprint. The template is deliberately incomplete (`complete=false`,
`count=0`, `shards=[]`). It is not a finished manifest and not a synthetic label set.
The following illustrates the older Git-commit provenance format; it is not the
accepted V2 manifest. Finalization requires successful independent verification:

```json
{
  "complete": true,
  "count": 149813,
  "generator_git_commit": "ACTUAL_GENERATOR_COMMIT",
  "oks_implementation_sha256": "SHA256_OF_SHARED_oks_utility.py",
  "shards": [
    {"file": "shard-00000.npz", "sha256": "ACTUAL_FILE_SHA256", "count": 1024}
  ]
}
```

The displayed shard is a format illustration, not a complete 149,813-person manifest.
V2 also accepts a verified `generator_source_snapshot` plus SHA256 provenance in
place of a generator Git commit. The original generator ran in a source snapshot
without Git metadata; do not invent a commit or alter its archived provenance.
Keep the template's other fields unchanged: schema, source cache fingerprint,
base checkpoint hash, annotation hash, alpha, joint order and gain definition.
Fingerprint uses `shared_cache.digest_json(cache.run)`, **not** raw run.json bytes.
File SHA256 uses `shared_cache.hash_file(path)`. All `.npz` files must be listed.
Use the Git/Linux LF version of shared Python sources when computing code hashes.
The WSL working copy uses LF; Windows CRLF checkout bytes produce different hashes
even when the code is identical. Do not edit cache provenance to bypass a mismatch.
N refuses incomplete labels, duplicate/missing people, wrong images, mixed cache
versions, changed alpha/order, checksum errors, nonfinite values or incorrect gain.
Y's audit should cover alpha=0/P0, full replacement, candidate independence,
coordinates, same-person GT, all-person coverage and representative visualizations.
N's parser cannot replace this independent semantic audit.

## Predictor v1 (a starting design, not a proven optimum)

Only these named tensors enter A:

- features [B,17,256], after GAU/before intervention; apply LayerNorm(256).
- uncertainty [B,17], the existing initial-distribution uncertainty.
- pose_coords [B,17,2] = (P0_coords - input_center) / input_scale.
- pose_scores [B,17], decoded P0 scores, distinct from initial uncertainty.

Uncertainty and score scalar means/stds are fitted on full **train only**, with a
1e-6 std floor, and stored as model buffers. Geometry uses the same crop normalization
on train and val, without GT keypoints or GT area. Do not clip signed gains.

Flatten each joint's (normalized x,y,normalized score) to 51 values. Context MLP:
51 -> 128 -> 64. Learned joint embedding: 17 types x 16 dimensions. Shared joint
regressor: (256 + 1 + 16 + 64)=337 -> 128 -> 64 -> 1, ReLU hidden activations,
linear output. Returns [B,17]. Argmax ties use lowest COCO index. No alpha classifier,
candidate acceptance, GT matching or actual candidate features are fed into A.

Default loss: SmoothL1(100 * predicted_gain, 100 * gain). Predictions and exported
values remain in raw OKS-gain units. Defaults: AdamW lr=1e-3, weight_decay=1e-4,
batch=256, 20 epochs, seed=42, full validation every 5 epochs and at the final epoch.
These are starting settings, not validated hyperparameters. Best checkpoint is
selected by full detection-val AP, not training loss. No training was run to pick them.

Train inputs stream through 32-shard buffers (about 1024 people for current shards).
Shuffle shard order and rows within each buffer each epoch; every person is used
once, including tails. This is bounded-buffer shuffling, not a global random
permutation. Only small target arrays are kept in memory; no full feature copy is
created in RAM/GPU or on disk. Resume is at completed epoch boundaries and rejects
changed code, data, settings or labels. Mid-epoch interruption repeats that epoch.

## Local WSL paths

Environment: `/home/nzd/miniconda3/envs/cigpose/bin/python`.
N working copy: `/home/nzd/projects/cig-module-a-n/code`.
Daily-use directory: `/home/nzd/projects/cig-module-a-n/code/Module_A`.
Original shared runs and data remain read-only inputs:

```text
train cache: /home/nzd/projects/cig-shared-cache-run-20261112/train2017
val cache:   /home/nzd/projects/cig-detection-public-20261116/val2017-detection
val GT:      /home/nzd/projects/CIGPose/data/coco/annotations/person_keypoints_val2017.json
base weights:/home/nzd/projects/CIGPose/checkpoints/cigpose-m_coco_256x192.pth
```

Commands below run from the N working copy root. Define task-specific variables:

```bash
A_PY=/home/nzd/miniconda3/envs/cigpose/bin/python
A_FYP=mmpose/analysis/fyp
A_TRAIN=/home/nzd/projects/cig-shared-cache-run-20261112/train2017
A_VAL=/home/nzd/projects/cig-detection-public-20261116/val2017-detection
A_ANN=/home/nzd/projects/CIGPose/data/coco/annotations/person_keypoints_val2017.json
A_BASE=/home/nzd/projects/CIGPose/checkpoints/cigpose-m_coco_256x192.pth
```

Safe interface-only commands (no labels generated, no optimizer steps):

```bash
$A_PY $A_FYP/tests/test_module_a.py
$A_PY $A_FYP/check_a_interface.py --train-cache "$A_TRAIN" --val-cache "$A_VAL" \
  --val-ann "$A_ANN" --base-checkpoint "$A_BASE" \
  --output /home/nzd/projects/cig-module-a-n/checks
```

The check produces `Y_metadata_template.json` and `interface_check.json`, no model
weights. Use a new output path when repeating checks. A standalone template can be
created with `a_dataset.py --train-cache "$A_TRAIN" --write-contract NEW_FILE.json`.

The full V2 labels have passed preflight. To repeat preflight (still no training):

```bash
A_LABELS=/home/nzd/projects/cig-module-a-n/labels/train2017_legacy_deterministic_v2_20261116
A_RUN=/home/nzd/projects/cig-module-a-n/runs/module_a_mlp_seed42
$A_PY Module_A/train.py --train-cache "$A_TRAIN" --labels "$A_LABELS" \
  --val-cache "$A_VAL" --val-ann "$A_ANN" --base-checkpoint "$A_BASE" --output "$A_RUN"
```

Only when training is explicitly intended, add `--run-training`. To resume the
same run, add both `--run-training --resume` with identical settings. This delivery
does not enable optimization by default. A stale `.training.lock` after process termination must
only be removed after verifying no process is writing.

For this machine, `config.py` already supplies these defaults. From `Module_A/`,
`python train.py` is sufficient for preflight; `python train.py --run-training`
starts optimization only when explicitly requested. The new trainer imports the
actual `Module_A/model.py`; it is not a launcher for the deleted trainer.

Train outputs: settings.json, history.json, last.pt, best.pt and full validation
folders. Best is selected among the configured validation epochs. Each val folder
contains raw_predictions_before_nms.json, selections.jsonl, predictions.keypoints.json,
metrics.json and evaluation.log. Both raw and final outputs feed existing M tools:
raw needs scoring/NMS once; final COCO JSON must not be postprocessed again.

Standalone full validation after obtaining a trained checkpoint:

```bash
$A_PY Module_A/evaluate.py --checkpoint "$A_RUN/best.pt" --output "$A_RUN/standalone-val"
```

The validation input has no annotation IDs. Export preserves source detection index,
full string instance ID, image ID, box, detector score and input_scale. All detections
are retained before standard postprocessing. Use full val annotations for evaluation,
never a GT-matched subset. Test-dev is outside this first implementation.
Standalone evaluation takes cache, annotation and base-model paths from
`Module_A/config.py`; update those paths when using a different machine.

## N/Y next handoff

1. Share the final precision profile and V2 label fingerprint with Y and B/C owners.
2. Use the accepted V2 labels; preserve Y's original archive and rejected V1 separately.
3. N's full-label preflight and real-data interface checks have passed; Y can assist
   with selection analysis and visualization without blocking training readiness.
4. First training is complete. Compare A-only against uncertainty Top-1 and cached
   P0 under the same V2 protocol using full-val AP; further training requires a new experiment.

Changing the A network alone does not require regenerating labels. Changing base
weights, candidate mechanism/alpha, crop inputs or reference definition does.
