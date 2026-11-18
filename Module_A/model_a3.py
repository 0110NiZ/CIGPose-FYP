"""Module A3: COCO-17 full-joint-context gain predictor.

Only joint selection is learned.
The frozen CIGPose and V2 signed-Gain labels remain unchanged.

A-only always chooses one joint with alpha=1.
Python 3.8 compatible.
"""

import common

import torch
from torch import nn

from a_contract import INPUT_KEYS


ARCHITECTURE = 'a3_full_joint_context_v1'

JOINTS = 17
FEATURE_DIM = 256

# 每个关节：256维特征 + 1维uncertainty + 2维坐标 + 1维置信度
# 总维度 = 17 * 260 = 4420
GLOBAL_DIM = JOINTS * (FEATURE_DIM + 1 + 2 + 1)


class MLP(nn.Module):

    def __init__(
        self,
        joint_dim=16,
        context_dim=64,
        hidden_dim=128,
        architecture=ARCHITECTURE
    ):
        super().__init__()

        if architecture != ARCHITECTURE:
            raise ValueError('Unsupported A3 architecture')

        if min(joint_dim, context_dim, hidden_dim) <= 0:
            raise ValueError('Model dimensions must be positive')

        self.config = dict(
            joint_dim=joint_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            architecture=architecture
        )

        # 对17个关节各自的256维特征进行归一化。
        self.feature_norm = nn.LayerNorm(FEATURE_DIM)

        # COCO-17关节身份编码。
        self.joint_embedding = nn.Embedding(
            JOINTS,
            joint_dim
        )

        # A3核心：
        # 将全部17个关节的信息按照COCO关节顺序展开。
        #
        # 不仅使用当前关节的特征，
        # 还让模型理解其他16个关节的完整状态。
        #
        # 4420 -> 128 -> 64
        self.context = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU()
        )

        # 对每个候选关节预测一个OKS Gain。
        #
        # 当前关节256维特征
        # + uncertainty
        # + 关节身份Embedding
        # + 全身Context
        #
        # 17个关节共享同一个收益预测器。
        self.regressor = nn.Sequential(
            nn.Linear(
                FEATURE_DIM + 1 + joint_dim + context_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

        # 训练集计算的归一化统计量。
        self.register_buffer(
            'scalar_mean',
            torch.zeros(2)
        )

        self.register_buffer(
            'scalar_std',
            torch.ones(2)
        )

        # Gain可以是正数，也可以是负数。
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
                'Invalid train-only scalar statistics'
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
        """
        输入：
            features:    [B, 17, 256]
            uncertainty: [B, 17]
            pose_coords: [B, 17, 2]
            pose_scores: [B, 17]

        输出：
            predicted_gain: [B, 17]
        """

        n = features.shape[0]

        if (
            features.shape != (n, JOINTS, FEATURE_DIM)
            or uncertainty.shape != (n, JOINTS)
            or pose_coords.shape != (n, JOINTS, 2)
            or pose_scores.shape != (n, JOINTS)
        ):
            raise ValueError(
                'Expected COCO-17 Module A input shapes'
            )

        # 全部17个关节的256维特征。
        f = self.feature_norm(features)

        # 全部17个关节的不确定性。
        u = (
            uncertainty - self.scalar_mean[0]
        ) / self.scalar_std[0]

        # 全部17个关节的置信度。
        s = (
            pose_scores - self.scalar_mean[1]
        ) / self.scalar_std[1]

        # 拼接全身信息：
        #
        # f:           [B,17,256]
        # u:           [B,17,1]
        # pose_coords: [B,17,2]
        # s:           [B,17,1]
        #
        # 得到 [B,17,260]，再展开为 [B,4420]。
        whole_person = torch.cat(
            (
                f,
                u.unsqueeze(-1),
                pose_coords,
                s.unsqueeze(-1)
            ),
            dim=-1
        ).reshape(n, GLOBAL_DIM)

        # 全身上下文：
        # [B,4420] -> [B,64]
        global_context = self.context(whole_person)

        # 关节身份编码。
        joint_ids = torch.arange(
            JOINTS,
            device=features.device
        )

        joint_identity = self.joint_embedding(
            joint_ids
        )

        joint_identity = joint_identity.unsqueeze(0).expand(
            n,
            -1,
            -1
        )

        # 让每个候选关节都能够看到全身上下文。
        global_context = global_context.unsqueeze(1).expand(
            -1,
            JOINTS,
            -1
        )

        # 当前关节局部信息 + 全身信息。
        joint_input = torch.cat(
            (
                f,
                u.unsqueeze(-1),
                joint_identity,
                global_context
            ),
            dim=-1
        )

        # 输出17个预测Gain。
        return self.regressor(
            joint_input
        ).squeeze(-1)


def to_device(inputs, device):

    if set(inputs) != set(INPUT_KEYS):
        raise ValueError(
            'Only approved Module A input fields are allowed'
        )

    return {
        key: torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device
        )
        for key, value in inputs.items()
    }


def select_joint(predicted_gain):
    """
    选择预测Gain最大的一个关节。

    即使17个Gain全部为负，
    A-only仍选择其中最大的一个。
    """

    if (
        predicted_gain.ndim != 2
        or predicted_gain.shape[1] != JOINTS
        or not torch.isfinite(predicted_gain).all()
    ):
        raise ValueError(
            'Expected finite predicted gains [B,17]'
        )

    return predicted_gain.argmax(dim=1)
