# Five-group experiment 4 (G4) — results and single-GPU reproduction

## Result

G4: teacher CIM/GNN initialization + backbone feature distillation; student CIM/GNN frozen in epochs 1–10, then unfrozen. Training completed **100 epochs**. Epoch 100 was evaluated separately because the original run stopped in COCO metric computation.

| Metric | Epoch 100 |
|---|---:|
| COCO AP | 70.4272 |
| AP50 | 90.6049 |
| AP75 | 77.4889 |
| AR | 73.3848 |
| Occluded joint mean error (pixels) | 17.0739 |

Highest evaluated AP: **epoch 100**. These are COCO AP *points* (0–100); the raw JSON stores AP in [0,1]. The existing Epoch 90 `best_AP.pth` was not the highest after independent Epoch 100 evaluation. Model hash: `9b1ef9e2958e5fe391daecb10909dab6de7d2aa725f74e11a7590b94861ca35f`.

FLOPs, inference parameter count and visible-joint error are **not yet verified for this G4 artifact**, so they are intentionally left as `null` in `summary.json`. Do not interpret training-only feature projection parameters as inference compute without checking the actual inference path.

## Artifacts

- `best.pth`: Epoch 100 evaluation/prediction checkpoint (optimizer/scheduler removed, retains `state_dict` and `ema_state_dict`)
- `epoch100_metrics.json`: independently evaluated Epoch 100 metrics
- `ap_history.json`: recorded evaluations at epochs 10–90 and **independently evaluated** epoch 100
- `loss_history.csv`, `feature_history.csv`: full 100-epoch loss and distillation tracking
- `loss_curve.png`, `ap_curve.png`: visualizations (AP chart includes epoch 100)
- `progress.json`, `train.log` if present: training traces

## Single-GPU reproduction from VS Code / Linux / WSL2

Use the existing `lightweight/five_groups/` single-GPU training source in the `轻量化` branch of `0110NiZ/CIGPose-FYP`, together with its `lightweight/group_c_kd/` and MMPose dependencies. It is **not** a standalone installable project. Follow the original `lightweight/README.md` for dependencies, COCO 30% subset, teacher model and dataset layout.

From the repository root, with the original working C-group Python environment activated:

```bash
CUDA_VISIBLE_DEVICES=0 python lightweight/five_groups/train.py --group 4 --smoke
CUDA_VISIBLE_DEVICES=0 python lightweight/five_groups/train.py --group 4 --teacher checkpoints/cigpose-m-coco-256x192.pth --work-dir work_dirs/g4_reproduction
```

`--smoke` runs a small real backward-pass check; omit it for formal 100-epoch training. Use a new, empty work directory. A previous run can be resumed using `--resume --work-dir SAME_DIR` when that folder contains the original training checkpoint files.

This is a **single-GPU reproduction entry point**, not a claim that the 100-epoch Kaggle run used only one GPU throughout: Kaggle used a dual-T4 launcher and single-GPU resumed segments. The reproduced metrics can vary with environment, run configuration and RNG state.

The training/validation data specification and evaluator settings are inherited from `lightweight/group_c_kd/config.py` via `common.py`. The generic source package's `train.py` retains the original single-GPU checkpoint frequency; the HF backup run's `train.py` added `cfg.default_hooks.checkpoint.interval = 1` to save every epoch. This difference does not change the student architecture or loss formula.

## Model provenance

Epoch 100 model came from the Hugging Face G4 backup checkpoint, SHA256 validated before export. HF backup: https://huggingface.co/datasets/YXX9/cigpose-g4-kaggle-backup (may require access). The saved `best.pth` is 58–59 MiB and has SHA256 `9b1ef9e2958e5fe391daecb10909dab6de7d2aa725f74e11a7590b94861ca35f`.

Before relying on `best.pth` for deployment, run a standalone load/evaluation check with the supplied model configuration and confirm that the EMA/non-EMA path matches your validation convention.
