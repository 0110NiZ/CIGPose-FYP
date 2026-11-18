"""A3 + weak expected selection regret, coefficient0.1; no architecture change.

L = SmoothL1(100*prediction,100*gain)
    + weight * mean(sum_j softmax(100*prediction/temperature)_j
                    * (max_k(100*gain_k)-100*gain_j)).
Temperature=1 in scaled gain units (0.01 in raw gain units).
Inference still selects exactly one argmax joint, alpha=1, always accepts.
"""
import torch
from train import run,smooth_l1

CHOICE_WEIGHT=.1
TEMPERATURE=1.
GAIN_SCALE=100.
LOSS_SETTINGS=dict(kind='smooth_l1_plus_expected_selection_regret',
    choice_weight=CHOICE_WEIGHT,temperature=TEMPERATURE,gain_scale=GAIN_SCALE,
    description='原SmoothL1 + 0.1 × 选择收益损失；A3网络不变。')

def selection_regret(pred,target):
    if pred.shape!=target.shape or pred.ndim!=2 or pred.shape[1]!=17:
        raise ValueError('Expected matching [batch,17] gain tensors')
    probability=torch.softmax(pred*GAIN_SCALE/TEMPERATURE,dim=1)
    scaled_gain=target*GAIN_SCALE
    gaps=scaled_gain.max(dim=1,keepdim=True).values-scaled_gain
    return (probability*gaps).sum(dim=1).mean()

def choice_loss(pred,target,weight=CHOICE_WEIGHT):
    return smooth_l1(pred,target)+weight*selection_regret(pred,target)

if __name__=='__main__':
    run('model_a3_gain_weak','A3 弱选择收益Loss（0.1）',.001,
        loss_fn=choice_loss,order='buffered',loss_settings=LOSS_SETTINGS)
