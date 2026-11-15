_base_ = ['./cigpose-m_1xb384-420e_coco-256x192.py']
model = dict(test_cfg=dict(flip_test=False))
val_dataloader = dict(dataset=dict(
    bbox_file='../data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
    filter_cfg=dict(bbox_score_thr=0.0)))
test_dataloader = val_dataloader
val_evaluator = dict(score_mode='bbox_keypoint', keypoint_score_thr=0.2,
                     nms_mode='oks_nms', nms_thr=0.9, format_only=False)
test_evaluator = val_evaluator
