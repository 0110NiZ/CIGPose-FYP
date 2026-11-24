# COCO 50% Adaptive KD: T=0.2 comparison note

This run completed 100 epochs on the fixed 50% COCO annotation subset. Its final and best validation coco/AP was 0.716972, compared with 0.715881 for the archived T=0.1 run.

The two saved five-batch diagnostic caches contain the same 640 sample IDs, transformed keypoints, and flip flags. However, all 640 image input tensors differ: the mean absolute pixel difference is 15.6711 and the maximum is 255. The diagnostic inputs were therefore not identical. This check does not establish whether image inputs differed across the full training runs.

Treat the AP difference as an observed comparison, not as evidence that changing T alone caused it. Model checkpoints and diagnostic cache files remain in AutoDL work_dirs and are not included in Git.
