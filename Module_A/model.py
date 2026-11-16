
"""Module A A1: Global-Context Gain Predictor.

A1 在 A0 的基础上增加全身关节上下文：
1. 对全部 17 个关节的 256 维 feature 使用共享编码器。
2. 对编码后的特征进行 mean pooling 和 max pooling。
3. 将全部 17 个关节的 uncertainty 加入全局上下文。
4. 每个关节结合自身信息和全局上下文预测 Gain。

输出仍然是 [B, 17]，A-only 仍然通过 argmax 选择一个关节。
"""

import common
import torch
from torch import nn

from a_contract import INPUT_KEYS


class MLP(nn.Module):

    def __init__(
        self,
        joint_dim=16,
        context_dim=64,
        hidden_dim=128,
        architecture='a0'
    ):
        super().__init__()

        if architecture not in ('a0', 'a1_global'):
            raise ValueError(
                "architecture must be 'a0' or 'a1_global'"
            )

        self.architecture = architecture

        self.config = dict(
            joint_dim=joint_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            architecture=architecture
        )

        # 每个关节原始 feature 为 256 维。
        self.feature_norm = nn.LayerNorm(256)

        # 17 个关节各自的身份 embedding。
        self.joint_embedding = nn.Embedding(17, joint_dim)

        # A1：增加全关节特征编码器。
        if architecture == 'a1_global':

            self.global_feature_encoder = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU()
            )

            # 全局上下文维度：
            # 全身坐标 + 置信度：51
            # 全部 uncertainty：17
            # 全身 feature mean pooling：64
            # 全身 feature max pooling：64
            context_input_dim = 51 + 17 + 64 + 64

        else:
            # A0 原始结构，保留旧 checkpoint 兼容性。
            context_input_dim = 51

        # 全局上下文编码器。
        self.context = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, context_dim),
            nn.ReLU()
        )

        # 每个关节共享同一个 Gain 回归器。
        self.regressor = nn.Sequential(
            nn.Linear(
                256 + 1 + joint_dim + context_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

        # uncertainty 和 confidence 的训练集标准化参数。
        self.register_buffer(
            'scalar_mean',
            torch.zeros(2)
        )

        self.register_buffer(
            'scalar_std',
            torch.ones(2)
        )

        # Gain 可以是正数或负数。
        nn.init.normal_(
            self.regressor[-1].weight,
            std=1e-3
        )

        nn.init.zeros_(
            self.regressor[-1].bias
        )

    def set_input_statistics(self, mean, std):

        mean = torch.as_tensor(
            mean,
            dtype=torch.float32,
            device=self.scalar_mean.device
        )

        std = torch.as_tensor(
            std,
            dtype=torch.float32,
            device=self.scalar_std.device
        )

        if (
            mean.shape != (2,)
            or std.shape != (2,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError(
                'Invalid train-only normalization statistics'
            )

        self.scalar_mean.copy_(mean)
        self.scalar_std.copy_(std)

    def forward(
        self,
        features,
        uncertainty,
        pose_coords,
        pose_scores
    ):

        n = features.shape[0]

        # 检查输入形状。
        if (
            features.shape != (n, 17, 256)
            or uncertainty.shape != (n, 17)
            or pose_coords.shape != (n, 17, 2)
            or pose_scores.shape != (n, 17)
        ):
            raise ValueError(
                'Module A expects COCO-17 input shapes'
            )

        # 1. 标准化 uncertainty 和 confidence。
        u = (
            uncertainty - self.scalar_mean[0]
        ) / self.scalar_std[0]

        s = (
            pose_scores - self.scalar_mean[1]
        ) / self.scalar_std[1]

        # 2. 原来的全身姿态上下文。
        # 17 个关节的 x、y、score，共 51 维。
        pose_context = torch.cat(
            [
                pose_coords,
                s.unsqueeze(-1)
            ],
            dim=-1
        ).reshape(n, 51)

        # 3. 对全部关节 feature 做 LayerNorm。
        normalized_features = self.feature_norm(
            features
        )

        # 4. A1：提取全身关节上下文。
        if self.architecture == 'a1_global':

            # [B,17,256] -> [B,17,64]
            encoded = self.global_feature_encoder(
                normalized_features
            )

            # 全局平均池化。
            feature_mean = encoded.mean(dim=1)

            # 全局最大池化。
            feature_max = encoded.max(dim=1).values

            # 汇总所有关节的信息。
            # [B,51] + [B,17] + [B,64] + [B,64]
            # = [B,196]
            global_context = torch.cat(
                [
                    pose_context,
                    u,
                    feature_mean,
                    feature_max
                ],
                dim=-1
            )

        else:

            # A0 原始上下文。
            global_context = pose_context

        # 5. 压缩全局上下文。
        # [B,196] -> [B,64]
        context = self.context(
            global_context
        )

        # 6. 每个关节的身份编码。
        joint_ids = torch.arange(
            17,
            device=features.device
        )

        joint_types = self.joint_embedding(
            joint_ids
        )

        # 7. 每个关节接收：
        # 自身 256 维 feature
        # 自身 uncertainty
        # 关节身份 embedding
        # 整个人体的全局上下文
        x = torch.cat(
            [
                normalized_features,

                u.unsqueeze(-1),

                joint_types.unsqueeze(0).expand(
                    n, -1, -1
                ),

                context.unsqueeze(1).expand(
                    -1, 17, -1
                )
            ],
            dim=-1
        )

        # 8. 输出 17 个 Gain。
        # 不使用 sigmoid 或 softmax。
        return self.regressor(x).squeeze(-1)


def to_device(inputs, device):

    if set(inputs) != set(INPUT_KEYS):
        raise ValueError(
            'Only the four approved model input fields are accepted'
        )

    return {
        k: torch.as_tensor(
            v,
            dtype=torch.float32,
            device=device
        )
        for k, v in inputs.items()
    }


def select_joint(predicted_gain):

    if (
        predicted_gain.ndim != 2
        or predicted_gain.shape[1] != 17
        or not torch.isfinite(predicted_gain).all()
    ):
        raise ValueError(
            'Expected finite predicted gains [B,17]'
        )

    # A-only 始终选择一个关节。
    # 即使 17 个 Gain 全为负数，也选择最大的那个。
    return predicted_gain.argmax(dim=1)
