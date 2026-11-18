"""Local 128->64; only learning rate changes to 0.0003."""
from train import run
if __name__=='__main__':run('model_local_low_lr','局部模型，学习率0.0003',.0003)
