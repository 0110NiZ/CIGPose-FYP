"""Local gain predictor with a smaller 64->32 scoring MLP."""
from torch import nn
from model_local import MLP as Local

class MLP(Local):
    def __init__(self):
        super().__init__()
        self.regressor=nn.Sequential(nn.Linear(340,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.normal_(self.regressor[-1].weight,std=1e-3)
        nn.init.zeros_(self.regressor[-1].bias)
