# Module A | True Gain Labels: Supplementary Review

## Provenance of P0 comparison

**Official P0:** `P_0_coords` from NIE's COCO train2017 GT-box cache.

**Rebuilt P0:** decode the original, *unmodified* `features` from the same NIE cache using TANG's shared `predict_from_joint_features` and MMPose `decode`, then restore original-image coordinates.

This is **NOT** a comparison with any separately saved 'TANG P0' dataset.

The official labels use **NIE cached P0** for `oks_before`; reconstructed P0 is for diagnostics only.

## Full-data checks

- COCO train2017 GT-box: **149,813 persons**, 4,682 shards, 17 candidates per person.
- Independent data audit: `Full coverage=True`, `Issues=0` (see original `validation_report.json`).
- The independent audit verifies schema/IDs/coverage/OKS-before/Gain arithmetic, not re-inference of all candidates.

## Full Train2017 Gain statistics

- Total candidate gains: 2,546,821
- Positive/negative/zero: 828,848 / 1,694,009 / 23,964
- Mean true Gain: -0.01014629
- People with at least one positive candidate: 146,003
- Mean offline oracle-best Gain: +0.01176467 (not a learned-model result)

## P0 difference statistics (all 4,682 logged shards)

- Weighted mean absolute coordinate difference: **0.00098815 px**
- Maximum absolute coordinate difference: **295.96160 px**
- Weighted mean absolute instance OKS difference: **0.000010692**
- Maximum absolute instance OKS difference: **0.121422735**
- Outlier threshold counts below refer to **shards with a large maximum difference**, not affected people or joints.
  - max > 0.1 px: 2090 source shards
  - max > 1.0 px: 546 source shards
  - max > 5.0 px: 102 source shards
  - max > 10.0 px: 67 source shards
  - max > 20.0 px: 33 source shards

## Re-run semantic evidence

- Example candidates: 6
- Recomputed candidate checks passed: 6
- Candidate/alpha0 semantic status: **PASS**
- Each example was re-inferred in its original full 32-person source batch; results were checked against saved OKS and Gain.
- These sample checks do not replace a full model re-inference audit.

## Unresolved technical question for NIE / TANG

Does the original NIE cache P0 use exactly the same SimCC logit precision, argmax decoding, unmodified feature path, and original-image restoration as the reconstruction?
Large outliers are not assumed to be adjacent-bin effects. The saved cache does not include the original NIE SimCC logits, so the exact cause cannot be proven from rebuilt logits alone.

## Artifacts

- `true_gain_summary.json`, `per_joint_gain.csv`, `true_gain_distribution.png`: full-data Gain statistics.
- `p0_difference_summary.json`, `p0_shard_max_*.png`: full-data shard-level difference statistics and plots.
- `p0_top_shards.csv`, `p0_top_oks_shards.csv`: shard-level outliers.
- `p0_outlier_cases.csv`, `p0_outliers/*.png`: individual largest-difference cases.
- `gain_cases.csv`, `gain_cases/*.png`: positive/negative gain samples; candidate predictions rechecked.
- `semantic_checks.json`: exact sample validation results and any issues.

### Image background

If COCO train2017 image files are not supplied, figures show original-image coordinate plots on a neutral background; they are not photographic overlays.
