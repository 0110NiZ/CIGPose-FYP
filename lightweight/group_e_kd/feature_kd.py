from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmpose.registry import HOOKS, MODELS

from c_tracking import preserve_rng
from kd import DEFAULT_TEACHER, KDStudent, load_teacher


@contextmanager
def capture_cim_input(cim):
    """Read the actual CIM input on the same forward used for SimCC KD."""
    features = []

    def capture(module, args):
        features.append(args[0])

    handle = cim.register_forward_pre_hook(capture)
    try:
        yield features
    finally:
        handle.remove()


def feature_distance(student, teacher, weights):
    if (student.shape != teacher.shape or student.shape[1:] != (17, 256)
            or student.shape[:2] != weights.shape):
        raise ValueError('Expected matching [B, 17, 256] features and [B, 17] weights')
    with torch.autocast(device_type=student.device.type, enabled=False):
        distance = F.mse_loss(student.float(), teacher.detach().float(), reduction="none").mean(dim=-1)
        valid = weights > 0
        return distance.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


@MODELS.register_module()
class FeatureKDStudent(KDStudent):
    def loss(self, inputs, data_samples):
        if not self.training:
            return super().loss(inputs, data_samples)
        with capture_cim_input(self._teacher.head.cim) as teacher_features:
            with capture_cim_input(self.head.cim) as student_features:
                losses = super().loss(inputs, data_samples)
        teacher_joints, = teacher_features
        student_joints, = student_features
        weights = torch.cat([s.gt_instance_labels.keypoint_weights for s in data_samples])
        distance = feature_distance(student_joints, teacher_joints, weights)
        losses['feature_raw'] = distance.detach()
        losses['loss_feature'] = self.train_cfg['feature_weight'] * distance
        return losses


@HOOKS.register_module()
class TransferAndFinetuneHook(Hook):
    def __init__(self, warm_epochs, lr_factor, teacher_path=DEFAULT_TEACHER):
        self.warm_epochs = warm_epochs
        self.lr_factor = lr_factor
        self.teacher_path = Path(teacher_path)

    def before_train(self, runner):
        student = runner.model
        with preserve_rng():
            teacher = load_teacher(next(student.parameters()).device, self.teacher_path)
        # Attach after EMA construction. The teacher is outside the student module tree.
        object.__setattr__(student, '_teacher', teacher)
        modules = [(student.head.cim, teacher.head.cim), (student.head.gcn, teacher.head.gcn)]
        if runner.iter == 0:
            for target, source in modules:
                target.load_state_dict(source.state_dict(), strict=True)

        transferred = {id(p) for module, _ in modules for p in module.parameters()}
        self.transfer_groups = []
        self.reference_group = None
        covered = set()
        for group in runner.optim_wrapper.optimizer.param_groups:
            ids = {id(p) for p in group['params']}
            if ids & transferred:
                if ids - transferred:
                    raise ValueError('Optimizer mixes transferred parameters with other layers')
                self.transfer_groups.append(group)
                covered.update(ids)
            elif self.reference_group is None and ids:
                self.reference_group = group
        if covered != transferred or self.reference_group is None:
            raise ValueError('Optimizer does not cover the expected student parameter groups')

    def before_train_epoch(self, runner):
        # C's first 1,000 warmup steps scale every current LR multiplicatively.
        # Its cosine schedule starts at epoch 51, well after the groups reunite.
        factor = self.lr_factor if runner.epoch < self.warm_epochs else 1.0
        reference_lr = self.reference_group['lr']
        for group in self.transfer_groups:
            group['lr'] = reference_lr * factor
        runner.logger.info(f'E epoch {runner.epoch + 1}: CIM/GNN lr={reference_lr * factor:.8g}, '
                           f'other lr={reference_lr:.8g}, multiplier={factor:g}')
