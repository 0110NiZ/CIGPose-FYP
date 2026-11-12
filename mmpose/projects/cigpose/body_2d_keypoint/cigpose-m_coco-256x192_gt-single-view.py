"""FYP controlled AP/AR evaluation: GT boxes and single-view inference.

Run tools/test.py from the mmpose directory with the agreed checkpoint.
This is an evaluation overlay; the original reproduction config is preserved.
"""

_base_ = ['./cigpose-m_1xb384-420e_coco-256x192.py']

model = dict(test_cfg=dict(flip_test=False))

# Explicitly disable detection-box files for BOTH entry points. Keep the full
# original COCO val2017 annotation file and inherited deterministic pipeline.
val_dataloader = dict(dataset=dict(bbox_file=None))
test_dataloader = dict(dataset=dict(bbox_file=None))

# Pin the existing CocoMetric defaults so all variants share scoring and NMS.
# GT bbox_score is 1.0; pose score is bbox_score times the mean decoded
# keypoint score strictly above 0.2 (or zero if none pass).
val_evaluator = dict(
    type='CocoMetric',
    score_mode='bbox_keypoint',
    keypoint_score_thr=0.2,
    nms_mode='oks_nms',
    nms_thr=0.9,
    format_only=False,
)
test_evaluator = dict(
    type='CocoMetric',
    score_mode='bbox_keypoint',
    keypoint_score_thr=0.2,
    nms_mode='oks_nms',
    nms_thr=0.9,
    format_only=False,
)
