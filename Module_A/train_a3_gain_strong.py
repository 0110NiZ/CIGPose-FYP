"""A3 + strong expected selection regret, coefficient0.5; same temperature1."""
from train import run
from train_a3_gain_weak import choice_loss,LOSS_SETTINGS as WEAK_SETTINGS

CHOICE_WEIGHT=.5
LOSS_SETTINGS=dict(WEAK_SETTINGS,choice_weight=CHOICE_WEIGHT,
    description='原SmoothL1 + 0.5 × 选择收益损失；A3网络不变。')

def strong_choice_loss(pred,target):
    return choice_loss(pred,target,weight=CHOICE_WEIGHT)

if __name__=='__main__':
    run('model_a3_gain_strong','A3 强选择收益Loss（0.5）',.001,
        loss_fn=strong_choice_loss,order='buffered',loss_settings=LOSS_SETTINGS)
