"""Output KD with a GT gate for labelled, occluded joints."""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from mmpose.registry import METRICS, MODELS

from kd import KDHead, OccludedJointError


@contextmanager
def capture_predictions(head):
    """Capture the first call to each classifier, the intervened prediction."""
    predictions = {}

    def capture(axis):
        def hook(module, inputs, output):
            predictions.setdefault(axis, output)
        return hook

    handles = [head.cls_gx.register_forward_hook(capture('x')),
               head.cls_gy.register_forward_hook(capture('y'))]
    try:
        yield predictions
    finally:
        for handle in handles:
            handle.remove()


def occlusion_mask(samples, device):
    masks = []
    for sample in samples:
        points = np.asarray(sample.metainfo['raw_ann_info']['keypoints']).reshape(-1, 3)
        mask = points[:, 2] == 1
        if sample.metainfo.get('flip', False):
            mask = mask[np.asarray(sample.metainfo['flip_indices'])]
        masks.append(mask)
    return torch.as_tensor(np.stack(masks), device=device)


@torch.no_grad()
def gate(student, teacher, samples, valid, split_ratio):
    gt = torch.cat([torch.as_tensor(s.gt_instances.transformed_keypoints,
                                    device=valid.device, dtype=torch.float32)
                    for s in samples])
    student_xy = torch.stack([scores.argmax(-1) for scores in student], dim=-1) / split_ratio
    teacher_xy = torch.stack([scores.argmax(-1) for scores in teacher], dim=-1) / split_ratio
    student_error = torch.linalg.vector_norm(student_xy - gt, dim=-1)
    teacher_error = torch.linalg.vector_norm(teacher_xy - gt, dim=-1)
    occluded = valid & occlusion_mask(samples, valid.device)
    passed = occluded & (teacher_error < student_error)
    filtered = occluded & ~passed

    stats = {'gate_eligible': occluded.sum(), 'gate_passed': passed.sum()}
    for key, mask in (('passed', passed), ('filtered', filtered)):
        stats[f'gate_{key}_teacher_error_sum'] = teacher_error.masked_fill(~mask, 0).sum()
        stats[f'gate_{key}_student_error_sum'] = student_error.masked_fill(~mask, 0).sum()
    return valid & ~filtered, stats


@MODELS.register_module()
class GatedKDHead(KDHead):
    def loss(self, feats, batch_data_samples, train_cfg=None, teacher_target=None):
        if teacher_target is None:
            return super().loss(feats, batch_data_samples, train_cfg=train_cfg)
        with capture_predictions(self) as outputs:
            losses = super().loss(feats, batch_data_samples, train_cfg=train_cfg)

        student = (outputs['x'], outputs['y'])
        valid = torch.cat([sample.gt_instance_labels.keypoint_weights
                           for sample in batch_data_samples]) > 0
        keep, stats = gate(student, teacher_target, batch_data_samples,
                           valid, self.simcc_split_ratio)
        with torch.autocast(device_type=valid.device.type, enabled=False):
            beta = 1.0 / train_cfg['kd_temperature']
            axes = [
                F.kl_div(F.log_softmax(logits.float() * beta, dim=-1),
                         F.softmax(target.detach().float() * beta, dim=-1),
                         reduction='none').sum(-1)
                for logits, target in zip(student, teacher_target)
            ]
            # Keep the E-group denominator to preserve other joints' KL scale.
            raw = ((axes[0] + axes[1]) * 0.5).masked_fill(~keep, 0).sum()
            raw = raw / valid.sum().clamp_min(1)
        losses.update(kd_raw=raw.detach(), loss_kd=train_cfg['kd_weight'] * raw)
        losses.update({name: value.detach().float() for name, value in stats.items()})
        return losses


@METRICS.register_module()
class VisibleJointError(OccludedJointError):
    default_prefix = 'visible'

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            ann = sample['raw_ann_info']
            gt = np.asarray(ann['keypoints']).reshape(17, 3)
            mask = gt[:, 2] == 2
            prediction = np.asarray(sample['pred_instances']['keypoints']).reshape(17, 2)
            errors = np.linalg.norm(prediction[mask] - gt[mask, :2], axis=-1)
            self.results.append((int(mask.sum()), errors.sum(),
                                 errors.sum() / np.sqrt(ann['area'])))
