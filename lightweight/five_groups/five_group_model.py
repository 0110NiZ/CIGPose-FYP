"""Feature distillation for the existing C KDHead. Evaluation follows the C path."""
import torch
from torch import nn

from kd import KDStudent
from mmpose.registry import MODELS
from objectives import capture_output, joint_cosine, spatial_cosine


@MODELS.register_module()
class FiveGroupStudent(KDStudent):
    def __init__(self, feature_mode, feature_weight, **kwargs):
        super().__init__(**kwargs)
        if feature_mode not in ('joint', 'backbone', 'none'):
            raise ValueError(feature_mode)
        self.feature_mode = feature_mode
        self.feature_weight = float(feature_weight)
        if feature_mode == 'backbone':
            # Layer construction on CPU must not change shared student initialization.
            with torch.random.fork_rng(devices=[]):
                self.feature_projection = nn.Conv2d(512, 768, 1, bias=False)

    def loss(self, inputs, data_samples):
        if not self.training or self.feature_mode == 'none':
            return super().loss(inputs, data_samples)

        teacher = self._teacher
        with torch.no_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
            teacher_maps = teacher.extract_feat(inputs.float())
            if self.feature_mode == 'joint':
                with capture_output(teacher.head.gau) as target_tokens:
                    teacher_logits = teacher.head(teacher_maps)
                target_feature, = target_tokens
            else:
                teacher_logits = teacher.head(teacher_maps)
                target_feature = teacher_maps[-1]

        student_maps = self.extract_feat(inputs)
        if self.feature_mode == 'joint':
            with capture_output(self.head.gau) as student_tokens:
                losses = self.head.loss(student_maps, data_samples, train_cfg=self.train_cfg,
                                        teacher_target=teacher_logits)
            tokens, = student_tokens
            weights = torch.cat([sample.gt_instance_labels.keypoint_weights
                                 for sample in data_samples], dim=0)
            raw_feature = joint_cosine(tokens, target_feature, weights)
        else:
            losses = self.head.loss(student_maps, data_samples, train_cfg=self.train_cfg,
                                    teacher_target=teacher_logits)
            raw_feature = spatial_cosine(self.feature_projection(student_maps[-1]), target_feature)

        losses['feature_raw'] = raw_feature.detach()
        losses['loss_feature'] = self.feature_weight * raw_feature
        return losses
