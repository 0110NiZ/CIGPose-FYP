"""Exact C encoder/classifier from mai/module-c-v2 f5cdf4b2d288774b43714e702a980d9a0a4d7564; original137d variant."""
from typing import Optional
import torch
from torch import nn

def encode_module_c_features(
    p0_coords: torch.Tensor,
    pc_coords: torch.Tensor,
    bboxes_xyxy: torch.Tensor,
    p0_scores: torch.Tensor,
    pc_scores: torch.Tensor,
    joint_ids: torch.Tensor,
    alphas: torch.Tensor,
    uncertainty: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode scale-normalized pose change & intervention metadata for Module C."""
    cx = (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) * 0.5
    cy = (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) * 0.5
    center = torch.stack([cx, cy], dim=-1).unsqueeze(1)
    w = torch.clamp(bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0], min=1.0)
    h = torch.clamp(bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1], min=1.0)
    scale = torch.maximum(w, h).view(-1, 1, 1)

    p0_norm = ((p0_coords - center) / scale).reshape(p0_coords.size(0), -1)
    delta_norm = ((pc_coords - p0_coords) / scale).reshape(p0_coords.size(0), -1)
    delta_scores = pc_scores - p0_scores
    joint_onehot = torch.nn.functional.one_hot(joint_ids.long(), num_classes=17).float()
    alpha_feat = alphas.float().view(-1, 1)

    parts = [p0_norm, delta_norm, p0_scores.float(), delta_scores.float(), joint_onehot, alpha_feat]
    if uncertainty is not None:
        parts.append(uncertainty.float())
    return torch.cat(parts, dim=-1)

class ModuleCAcceptanceClassifier(nn.Module):
    """Lightweight binary classifier predicting P(gain > 0)."""

    def __init__(self, in_dim: int = 120, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
