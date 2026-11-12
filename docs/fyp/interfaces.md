# FYP Shared Experimental Protocol and Interfaces

This document defines the shared experimental settings, data fields, and common interfaces used by Modules A, B, and C.

## 1. Experimental Protocol

### 1.1 Base model

- Base model: CIGPose-m
- Dataset keypoints: COCO 17-keypoint format
- Input resolution: 256 x 192 (height x width)
- The base CIGPose pose-estimation components remain frozen during training of the new decision modules.
- The frozen base model must remain in evaluation mode to avoid updates to BatchNorm statistics.
- Shared source config: `mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py`
- Shared CIGPose checkpoint: `checkpoints/cigpose-m_coco_256x192.pth`
- Checkpoint SHA-256: `3dd69c8ab4e3c7a923a0183fd9abf844cb7ef13d39d26693baa082ca7931b3d0`
- The repository checkpoint path is a symbolic link on the current development machine; the SHA-256 identifies the actual weight file independently of its local storage path.

### 1.2 Data split

- Module training and development use the fixed image-level split of COCO train2017 prepared by LIN.
- Train and internal-dev images must not overlap.
- All person instances from the same source image must remain in the same subset.
- The fixed split files and valid-person annotation lists prepared by LIN are reused by all modules.
- COCO val2017 is not used for hyperparameter tuning or model selection unless the team explicitly revises the protocol and records the change.

### 1.3 Reference pose and gain definition

- Instance-level OKS is the formal supervision metric for Modules A, B, and C.
- P_0 is the non-intervention reference pose.
- P_0 is generated from the original, unmodified keypoint features through the same downstream graph reasoning and decoding used for intervention candidates.
- P_0 is not the original final output of CIGPose because the original CIGPose output includes feature intervention.
- P_c is a candidate pose generated after applying a specified keypoint intervention and then using the same downstream graph reasoning and decoding.

Refinement gain is defined as:

gain = OKS(P_c, GT) - OKS(P_0, GT)

- Positive gain means that the candidate improves the complete pose.
- Negative gain means that the candidate makes the complete pose worse.
- GT, true OKS, and true gain are used only for supervision, development validation, and offline analysis.
- GT, true OKS, and true gain must never be used as inference inputs to Modules A, B, or C.
- The shared OKS implementation maintained by LIN is used throughout the project.

### 1.4 Intervention convention

- Keypoint indices use the COCO 0-16 order.
- alpha = 0 represents no intervention and must reproduce P_0 under identical downstream processing.
- alpha = 1 represents full feature replacement.
- Module B predicts only the four non-zero candidate strengths:

  {0.25, 0.50, 0.75, 1.00}

- alpha = 0 is supported by the common intervention interface for validation and reference-pose checks, but it is not a Module B prediction class.

### 1.5 Evaluation consistency

The following settings must remain consistent within each controlled comparison:

- person bounding-box source
- input resolution
- base CIGPose checkpoint
- preprocessing
- flip_protocol
- flip_test
- coordinate restoration
- decoding
- pose-score computation
- post-processing / NMS

Any change to these settings must be recorded with the experiment.

### 1.6 Fixed flip protocol

The main experimental protocol uses single-view inference.

- `flip_protocol: single_view`
- `flip_test: false`
- No test-time horizontally flipped view is generated.
- No original/flip prediction fusion is performed.
- This rule applies to shared-cache generation, gain-label generation, Module A/B/C validation, and the formal AP/AR controlled comparisons.
- Original CIGPose, P_0, Uncertainty Top-1, single-module configurations, pairwise configurations, and A+B+C must use the same single-view protocol within the main comparison.
- Previously reproduced official CIGPose results obtained with `flip_test=True` may be retained as separate reproduction records, but they must not be used as the matched baseline for the single-view main experiments.
- A separate original-CIGPose single-view baseline must be evaluated for the main controlled comparison.

Single-view refers only to test-time flip augmentation. It does not prohibit training augmentation.

For the first version of shared cache generation:
- preprocessing must be deterministic
- random horizontal flipping must not be applied during cache generation
- any later training augmentation policy must be separately documented and recorded

### 1.7 Reference and candidate forward behaviour

- P_0 starts from the untouched keypoint features.
- P_0 performs no keypoint feature replacement.
- P_0 still passes through the same downstream GNN, decoding, scoring, and coordinate-restoration path used by candidate poses.
- P_c starts from the same untouched reference features as P_0.
- Only the specified joint is modified according to the specified alpha.
- Every candidate is independently generated from the untouched reference state.
- Candidate generation must never continue from features modified by a previous candidate.
- Different persons in the same batch may use different joint IDs and different alpha values.
- The frozen base CIGPose model must remain in evaluation mode.

## 2. Shared Data Schema

### 2.1 Instance identity

Each person instance should include:

- dataset_split
- image_id
- instance_id
- annotation_id, when GT pairing is available

Instance identity must not depend on DataLoader order.

### 2.2 Bounding-box and coordinate metadata

The shared metadata should include:

- bbox
- bbox_format
- bbox_score
- input_center
- input_scale
- input_size

The information required to restore predictions to original-image coordinates must be retained.

### 2.3 Shared model state

The common forward interface should expose:

- features: [B, 17, 256]
- uncertainty: [B, 17]

The shared keypoint features are taken after the GAU stage and before feature intervention.

Predictive uncertainty is the uncertainty used by the original CIGPose keypoint-selection procedure. It must be kept separate from final decoded keypoint scores.

### 2.4 Reference pose

The reference output should include:

- P_0_coords: [B, 17, 2]
- P_0_scores: [B, 17]

P_0 coordinates used for OKS calculation must be restored to the original-image coordinate system.

### 2.5 Intervention action

The common intervention interface should accept:

- joint_ids: [B]
- alphas: [B]

Different person instances in the same batch must be allowed to use different joint IDs and different intervention strengths.

### 2.6 Candidate pose

The candidate output should include:

- P_c_coords: [B, 17, 2]
- P_c_scores: [B, 17]

P_c and P_0 must:
- correspond to the same person instance
- use the same coordinate system
- use the same downstream graph reasoning and decoding procedure

### 2.7 Offline supervision fields

Offline supervision records should include at least:

- image_id
- annotation_id
- joint_id
- alpha
- gt_keypoints
- gt_area
- oks_before
- oks_after
- gain

These fields are used for supervision and offline analysis only.

## 3. Required Common Interfaces

### 3.1 Reference-state extraction

Expected behaviour:

```python
state = extract_reference(batch)
```

The returned state should provide:

- features
- uncertainty
- P_0 coordinates
- P_0 scores
- instance identity
- bounding-box and coordinate-restoration metadata

### 3.2 Candidate generation

Expected behaviour:

```python
candidate = generate_candidate(
    state,
    joint_ids,
    alphas,
)
```

Requirements:

- every candidate must be generated from the same untouched reference features
- the reference features must not be modified in-place
- different persons in one batch may use different joint IDs
- different persons in one batch may use different intervention strengths
- alpha = 0 must reproduce P_0
- alpha = 1 must match full replacement for the same selected joint
- candidate generation must use the same downstream graph reasoning and decoding as P_0

### 3.3 Combined-module data consistency

For combined configurations such as A+B, A+C, B+C, and A+B+C:

- downstream training data must be generated using the actual decisions of the corresponding fixed upstream module
- training-time candidate construction should match the inference-time decision process
- oracle or ground-truth decisions may be used only for offline reference analysis
- oracle or ground-truth decisions must not be used as actual inference inputs

### 3.4 Interface acceptance checks

The common interface must be verified on real person instances before large-scale cache generation.

Required checks:

- With alpha = 0, P_c must reproduce P_0 within numerical tolerance and the resulting OKS gain must be approximately zero.
- With alpha = 1, P_c must match full feature replacement for the same selected joint.
- The P_0 path must match the equivalent no-intervention single-view reference computation.
- Under the same joint and alpha, candidate generation must match the equivalent forced-intervention single-view computation.
- Original-image coordinates must be restored correctly.
- Person identity, image identity, annotation identity when available, bounding boxes, and prediction scores must remain correctly aligned.
- Batch-wise joint IDs and alpha values must act independently for different person instances.

## 4. Reproducibility Metadata

Each experiment or shared cache should record:

- schema_version
- git_commit
- config path or config hash
- checkpoint path or checkpoint hash
- split version
- valid-person-list version
- random seed
- flip_protocol
- flip_test
- numeric precision
- output location

## 5. Items Requiring Team Confirmation

The following items must be confirmed before large-scale data generation:

- bounding-box source for formal AP/AR evaluation
- final cache serialization format
- exact naming convention for shared cache files
- whether any later protocol change from the submitted Proposal is approved by the full team
