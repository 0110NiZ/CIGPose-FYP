"""A0 original pose-context MLP, SmoothL1."""
from train import run
if __name__=='__main__':run('model_a0','A0 姿态上下文',.001,order='buffered')
