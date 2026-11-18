"""Direct whole-person MLP without uncertainty; original A3 training settings."""
from train import run

if __name__=='__main__':
    run('model_direct_without_u','直接MLP：不带不确定性（4403→128→64→17）',.001,order='buffered')
