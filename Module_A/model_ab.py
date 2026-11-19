"""Frozen original A3 + trainable B, matching B f1df444's 563-128-64-4 MLP."""
import numpy as np
import torch
from torch import nn

ALPHAS = (0.25, 0.50, 0.75, 1.00)
GAIN_SCALE = 1000.0
ARCHITECTURE = 'ab_frozen_a3_b563_v1'


def make_features(arrays, joint_ids):
    """Explicit A-selected joints; never overwrite the cache's top1_joint."""
    joint = np.asarray(joint_ids)
    f = np.asarray(arrays['features'])
    n = len(f)
    if f.shape != (n, 17, 256) or joint.shape != (n,):
        raise ValueError('Expected features [N,17,256] and joint_ids [N]')
    if not np.issubdtype(joint.dtype, np.integer) or np.any((joint < 0) | (joint >= 17)):
        raise ValueError('Joint IDs must be integers in 0..16')
    x = np.concatenate((f[np.arange(n), joint], f.mean(1),
                        arrays['uncertainty'], arrays['confidence'],
                        np.eye(17, dtype=np.float32)[joint]), axis=1).astype(np.float32)
    if x.shape != (n, 563) or not np.isfinite(x).all():
        raise ValueError('Invalid B inputs')
    return x


class BRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(563, 128), nn.ReLU(), nn.Dropout(.1),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
        self.register_buffer('mean', torch.zeros(563))
        self.register_buffer('std', torch.ones(563))

    def forward(self, x):
        # Outputs are scaled gains, matching the original B training target.
        return self.network((x - self.mean) / self.std)

    def set_statistics(self, x):
        mean = x.mean(0, dtype=np.float64).astype(np.float32)
        std = x.std(0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.
        self.mean.copy_(torch.as_tensor(mean, device=self.mean.device))
        self.std.copy_(torch.as_tensor(std, device=self.std.device))


def choose_alpha(prediction):
    if prediction.ndim != 2 or prediction.shape[1] != 4 or not torch.isfinite(prediction).all():
        raise ValueError('Expected finite B output [N,4]')
    # Exact ties select the smallest alpha. All-negative outputs still select one.
    return prediction.new_tensor(ALPHAS)[prediction.argmax(1)]
