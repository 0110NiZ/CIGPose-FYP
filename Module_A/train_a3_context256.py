"""A3 summary width 256; original SmoothL1 and A3 training settings."""
from train import run

if __name__=='__main__':
    run('model_a3_context256','A3 全身摘要256',.001,order='buffered')
