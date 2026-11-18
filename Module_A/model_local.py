"""Local 128->64 gain predictor. No extra cross-joint input in Module A.

The frozen inherited context weights and zero slots preserve exact compatibility
with the completed experiments; they do not affect predictions or optimization.
"""
import torch
from torch import nn
from model_a3 import MLP as A3

class MLP(A3):
    def __init__(self):
        super().__init__()
        self.regressor[0]=nn.Linear(340,128)
        self.context.requires_grad_(False)

    def forward(self,features,uncertainty,pose_coords,pose_scores):
        n=len(features)
        f=self.feature_norm(features)
        u=(uncertainty-self.scalar_mean[0])/self.scalar_std[0]
        s=(pose_scores-self.scalar_mean[1])/self.scalar_std[1]
        identity=self.joint_embedding(torch.arange(17,device=f.device))[None].expand(n,-1,-1)
        x=torch.cat((f,u[...,None],identity,f.new_zeros(n,17,64),pose_coords,s[...,None]),-1)
        return self.regressor(x).squeeze(-1)
