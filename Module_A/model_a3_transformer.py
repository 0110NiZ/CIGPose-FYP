"""Small joint Transformer replaces A3's context compressor and scorer.

Exactly the same four cached input fields and frozen CIGPose are used.
Each of the 17 joint tokens contains normalized feature, uncertainty,
coordinates, confidence and a learned joint identity. Two independent
pre-norm encoder layers exchange information WITHIN each person only.
Output remains 17 signed predicted gains; inference selects one argmax.
No dropout, matching A3; no new loss or intervention policy.
"""
import common
import torch
from torch import nn
from model_a3 import MLP as A3, JOINTS, FEATURE_DIM

ARCHITECTURE='a3_joint_transformer_small_v1'

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=dict(architecture=ARCHITECTURE,joint_dim=16,token_dim=128,
                         layers=2,heads=4,feedforward_dim=256,dropout=0.,
                         norm_first=True,activation='gelu',scorer_hidden=64)
        self.feature_norm=nn.LayerNorm(FEATURE_DIM)
        self.joint_embedding=nn.Embedding(JOINTS,16)
        self.input_projection=nn.Linear(FEATURE_DIM+1+2+1+16,128)
        # Construct separately: avoid cloned layers with identical initialization.
        self.encoder=nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=128,nhead=4,dim_feedforward=256,
                dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            for _ in range(2)])
        self.output_norm=nn.LayerNorm(128)
        self.regressor=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.register_buffer('scalar_mean',torch.zeros(2))
        self.register_buffer('scalar_std',torch.ones(2))
        nn.init.normal_(self.regressor[-1].weight,std=1e-3)
        nn.init.zeros_(self.regressor[-1].bias)

    # Keep exactly the same train-only statistics and validation rules as A3.
    set_input_statistics=A3.set_input_statistics

    def forward(self,features,uncertainty,pose_coords,pose_scores):
        n=features.shape[0]
        if (features.shape!=(n,JOINTS,FEATURE_DIM)
            or uncertainty.shape!=(n,JOINTS)
            or pose_coords.shape!=(n,JOINTS,2)
            or pose_scores.shape!=(n,JOINTS)):
            raise ValueError('Expected COCO-17 Module A input shapes')
        f=self.feature_norm(features)
        u=(uncertainty-self.scalar_mean[0])/self.scalar_std[0]
        s=(pose_scores-self.scalar_mean[1])/self.scalar_std[1]
        identity=self.joint_embedding(torch.arange(JOINTS,device=features.device))
        identity=identity.unsqueeze(0).expand(n,-1,-1)
        tokens=torch.cat((f,u.unsqueeze(-1),pose_coords,s.unsqueeze(-1),identity),dim=-1)
        tokens=self.input_projection(tokens)
        for layer in self.encoder:
            tokens=layer(tokens)
        return self.regressor(self.output_norm(tokens)).squeeze(-1)
