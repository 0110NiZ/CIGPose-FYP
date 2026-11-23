import os.path as osp
_base_ = ['./team128_baseline100.py']

train_dataloader = dict(
    dataset=dict(
        ann_file=osp.abspath('../lightweight/data/person_keypoints_train2017_stratified50_nested30_seed21.json')
    )
)
