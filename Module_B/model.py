"""Small MLP for four intervention gains. CIGPose stays frozen."""
from Module_B.config import FEATURE_SPEC, ALPHAS


def get_model(device='cuda:0'):
    from torch import nn
    return nn.Sequential(
        nn.Linear(FEATURE_SPEC['dimension'], 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, len(ALPHAS)),
    ).to(device)
