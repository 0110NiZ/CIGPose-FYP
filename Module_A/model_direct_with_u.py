"""Fixed-order whole-person MLP: 4420 -> 128 -> 64 -> 17.

Same A3 feature/scalar normalization; no joint embedding, local branch,
or shared per-joint scorer. Output columns follow the fixed COCO-17 order.
The no-uncertainty variant reuses this implementation with 4403 inputs.
"""
import common
import torch
from torch import nn
from model_a3 import MLP as A3, JOINTS, FEATURE_DIM

class MLP(nn.Module):
    def __init__(self,include_uncertainty=True):
        super().__init__()
        self.include_uncertainty=bool(include_uncertainty)
        self.input_dim=JOINTS*(FEATURE_DIM+2+1+int(self.include_uncertainty))
        self.config=dict(architecture='direct_whole_person_mlp_v1',
            include_uncertainty=self.include_uncertainty,input_dim=self.input_dim,
            hidden_dims=[128,64],output_dim=JOINTS)
        self.feature_norm=nn.LayerNorm(FEATURE_DIM)
        self.regressor=nn.Sequential(nn.Linear(self.input_dim,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),nn.Linear(64,JOINTS))
        # Preserve the common train-only statistics interface. The no-u model
        # never uses scalar_mean[0]/scalar_std[0] or uncertainty values.
        self.register_buffer('scalar_mean',torch.zeros(2))
        self.register_buffer('scalar_std',torch.ones(2))
        nn.init.normal_(self.regressor[-1].weight,std=1e-3)
        nn.init.zeros_(self.regressor[-1].bias)

    set_input_statistics=A3.set_input_statistics

    def forward(self,features,uncertainty,pose_coords,pose_scores):
        n=features.shape[0]
        if (features.shape!=(n,JOINTS,FEATURE_DIM)
            or uncertainty.shape!=(n,JOINTS)
            or pose_coords.shape!=(n,JOINTS,2)
            or pose_scores.shape!=(n,JOINTS)):
            raise ValueError('Expected COCO-17 Module A input shapes')
        fields=[self.feature_norm(features)]
        if self.include_uncertainty:
            u=(uncertainty-self.scalar_mean[0])/self.scalar_std[0]
            fields.append(u.unsqueeze(-1))
        s=(pose_scores-self.scalar_mean[1])/self.scalar_std[1]
        fields.extend([pose_coords,s.unsqueeze(-1)])
        whole_person=torch.cat(fields,dim=-1).reshape(n,self.input_dim)
        return self.regressor(whole_person)
