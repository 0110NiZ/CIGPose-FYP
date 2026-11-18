"""Direct whole-person MLP with uncertainty; original A3 training settings."""
from train import run

if __name__=='__main__':
    run('model_direct_with_u','直接MLP：带不确定性（4420→128→64→17）',.001,order='buffered')
