# Group 3 - Backbone Feature Distillation

## Experimental Setup

- Dataset: COCO train2017, fixed 30% subset
- Validation: COCO val2017
- Training: 100 epochs
- Teacher: Official CIGPose-m (frozen)
- Student: Lightweight CIGPose
- Initialization: Teacher CIM/GNN weights
- Feature KD: Backbone feature cosine alignment
- Projection: 1x1 Conv, 512 -> 768 channels
- CIM/GNN learning rate: 0.1x for epochs 1-10, then 1x
- SimCC output KD retained

## Final Results (Epoch 100)

| Metric | Value |
|---|---:|
| COCO AP | 70.7873 |
| AP50 | 90.5889 |
| AP75 | 78.5352 |
| Occluded Mean Error | 17.1725 px |

## Comparison

Student baseline (A):
- AP: 70.3364
- Occluded error: 17.633 px

Group 3:
- AP improvement: +0.4509
- Occluded error reduction: approximately 0.4605 px

## Notes

The experiment completed 100 epochs.
Epoch 100 achieved the best validation AP.
The checkpoint is stored locally in work_dirs/five_group_3/.
Large checkpoints and training logs are not included in Git.
