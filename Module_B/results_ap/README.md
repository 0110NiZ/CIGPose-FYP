# Module B aligned validation results

All 149813 approved train2017 people train B. No internal development holdout.
Full val2017 COCO AP selects the best epoch. Exact ties retain the earlier epoch.
This is selected validation performance, not an independent test score.

| Method | AP |
| --- | ---: |
| P0 | 73.6223 |
| Fixed alpha=0.25 | 73.6652 |
| Top-1 alpha=1 | 73.5402 |
| B | 73.6726 |

B minus controls (AP points):
- P0: +0.050329
- Fixed alpha=0.25: +0.007440
- Top-1 alpha=1: +0.132409

All ten metrics and postprocessing records match the public evaluator.
The original internal-development experiment is retained separately in ../results.
Large caches, gain shards and all epoch checkpoints stay on the experiment drive.
