"""A3 summary width 128; original SmoothL1 and A3 training settings."""
from train import run

if __name__=='__main__':
    run('model_a3_context128','A3 全身摘要128',.001,order='buffered')
