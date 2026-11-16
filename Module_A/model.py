"""A 的小 MLP：输入人体信息，输出选 17 个关节各自可能带来的收益。"""
import common
import torch
from torch import nn

from a_contract import INPUT_KEYS


class MLP(nn.Module):
    def __init__(self, joint_dim=16, context_dim=64, hidden_dim=128):
        super().__init__()
        self.config = dict(joint_dim=joint_dim, context_dim=context_dim, hidden_dim=hidden_dim)
        # 1. 每个关节有 256 维特征；关节编号也做一个可学习编码。
        self.feature_norm = nn.LayerNorm(256)
        self.joint_embedding = nn.Embedding(17, joint_dim)
        # 2. 全身 17 个关节的 x、y、分数，共 51 个数，压成 64 维上下文。
        self.context = nn.Sequential(nn.Linear(51, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, context_dim), nn.ReLU())
        # 3. 当前关节的特征 + uncertainty + 关节身份 + 全身上下文，预测一个 Gain。
        self.regressor = nn.Sequential(
            nn.Linear(256 + 1 + joint_dim + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer('scalar_mean', torch.zeros(2))
        self.register_buffer('scalar_std', torch.ones(2))
        # 输出初始接近 0；Gain 可以为正或负。
        nn.init.normal_(self.regressor[-1].weight, std=1e-3)
        nn.init.zeros_(self.regressor[-1].bias)

    def set_input_statistics(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.scalar_mean.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.scalar_std.device)
        if mean.shape != (2,) or std.shape != (2,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid train-only normalization statistics')
        self.scalar_mean.copy_(mean)
        self.scalar_std.copy_(std)

    def forward(self, features, uncertainty, pose_coords, pose_scores):
        n = features.shape[0]
        if (features.shape != (n, 17, 256) or uncertainty.shape != (n, 17)
                or pose_coords.shape != (n, 17, 2) or pose_scores.shape != (n, 17)):
            raise ValueError('Module A expects COCO-17 input shapes')
        # 用 train 的统计量标准化 uncertainty 和关键点分数。
        u = (uncertainty - self.scalar_mean[0]) / self.scalar_std[0]
        s = (pose_scores - self.scalar_mean[1]) / self.scalar_std[1]
        context = self.context(torch.cat([pose_coords, s[..., None]], dim=-1).reshape(n, 51))
        types = self.joint_embedding(torch.arange(17, device=features.device))
        x = torch.cat([self.feature_norm(features), u[..., None],
                       types[None].expand(n, -1, -1), context[:, None].expand(-1, 17, -1)], dim=-1)
        # 最后一层不加 sigmoid/softmax，保留负收益；结果是 [人数, 17]。
        return self.regressor(x).squeeze(-1)


def to_device(inputs, device):
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError('Only the four approved model input fields are accepted')
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in inputs.items()}


def select_joint(predicted_gain):
    if predicted_gain.ndim != 2 or predicted_gain.shape[1] != 17 or not torch.isfinite(predicted_gain).all():
        raise ValueError('Expected finite predicted gains [B,17]')
    # A-only 总选一个关节；全负也选最大值。是否回退由 C 处理。
    return predicted_gain.argmax(dim=1)
