# Module A: Train2017 True Gain Statistics

## Dataset

- Dataset: COCO Train2017, GT-box
- Persons: 149,813
- Shards: 4,682
- Gain values: 2,546,821
- Candidates per person: 17
- Intervention alpha: 1
- Official baseline: NIE cached P_0_coords

## Gain Distribution

- Positive: 828,848 (32.54%)
- Negative: 1,694,009 (66.51%)
- Zero: 23,964 (0.94%)

## Gain Statistics

- mean: -0.010146
- std: 0.033200
- min: -0.999813
- p01: -0.130905
- p05: -0.057819
- p25: -0.008155
- median: -0.000811
- p75: 0.000083
- p95: 0.005088
- p99: 0.023044
- max: 0.998587

## Person-Level Analysis

- Persons with at least one positive candidate: 146,003
- Percentage with a positive candidate: 97.46%
- Persons whose best gain is zero: 315
- Persons whose best gain is negative: 3,495

## OKS Comparison

- Mean NIE P0 OKS: 0.891500
- Mean candidate OKS: 0.881354
- Mean oracle-best candidate OKS: 0.903265
- Mean oracle-best gain: 0.011765

## Interpretation Notes

- True Gain is computed using the NIE cached P0.
- Rebuilt P0 is a separate numerical diagnostic.
- Rebuilt P0 is obtained from NIE cached original features through TANG's public prediction/decoding interfaces.
- P0 diagnostics do not compare NIE P0 against an independently saved TANG P0.
- Oracle-best Gain is an offline label analysis, not the performance of a trained Gain Predictor.
- These statistics do not represent COCO val AP/AR.

## Validation

- Full coverage: True
- Independent audit passed: True
- Validation issues: 0
- Metadata complete: False

