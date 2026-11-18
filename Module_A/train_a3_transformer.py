"""Small joint Transformer; original SmoothL1 and A3 training settings."""
from train import run

if __name__=='__main__':
    run('model_a3_transformer','A3 小型Transformer（2层128维4头）',.001,order='buffered')
