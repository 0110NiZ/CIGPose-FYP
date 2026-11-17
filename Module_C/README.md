# Module C: Acceptance / Fallback Classifier

Module C decides whether to accept the fixed Top-1, alpha=1 candidate pose
or fall back to the no-intervention reference pose P0.

## Target

For each training person:

- candidate: fixed Top-1 joint, alpha=1
- gain = OKS(Pc, GT) - OKS(P0, GT)
- label = 1 if gain > 0, otherwise 0

GT gain is used only for offline supervision and analysis.
It is not an inference input.

## Input features

The classifier uses:

- normalized P0 coordinates
- normalized Pc - P0 coordinate changes
- P0 keypoint scores
- Pc - P0 score changes
- selected-joint one-hot encoding
- intervention strength alpha
- uncertainty features

The final feature dimension is 137.

## Training

`train.py` is the ordinary BCE baseline.

`train_weighted.py` uses gain-magnitude weighted BCE.
The binary target is unchanged, but samples with near-zero gain receive
smaller training weight.

Final weighting scale:

    gain_scale = 0.001

All experiments use seed 42 and the shared deterministic runtime policy.

## Validation protocol

Validation uses the frozen val2017 detection cache:

- 104,125 detections
- fixed detection boxes
- single-view inference
- Top-1 + alpha=1 candidate
- P0 fallback
- COCO AP used for model and threshold selection

The acceptance rule is:

    accept Pc if p_acc > tau
    otherwise use P0

## Final val2017 configuration

Final selected configuration:

- training: gain-weighted BCE
- epoch: 14
- tau: 0.51
- AP: 0.736832939710
- AR: 0.790380982368
- accept rate: 0.370756
- harmful elimination rate: 0.826016

Reference AP:

- Top-1, alpha=1: 0.735402167532
- P0: 0.736222992815

The final epoch and threshold are selected only on val2017.
They must be frozen before independent test-dev2017 evaluation.
