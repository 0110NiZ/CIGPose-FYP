from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS, MODELS
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.models.heads.coord_cls_heads.cig_head import CIGHead
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.utils.tensor_utils import to_numpy

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "CIGPose/mmpose"
TEACHER_CFG = REPO / "projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py"


def load_teacher(device):
    cfg = Config.fromfile(str(TEACHER_CFG))
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(HERE / "teacher.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).requires_grad_(False).eval()


def simcc_kd(student, teacher, weights):
    # Both models use beta=10 to turn final SimCC scores into distributions.
    # x and y have different numbers of bins: sum each KL over bins first.
    valid = (weights > 0).float()
    with torch.autocast("cuda", enabled=False):
        axes = []
        for logits, target in zip(student, teacher):
            assert logits.shape == target.shape
            log_p = F.log_softmax(logits.float() * 10., dim=-1)
            q = F.softmax(target.detach().float() * 10., dim=-1)
            axes.append(F.kl_div(log_p, q, reduction="none").sum(-1))
        return (((axes[0] + axes[1]) * 0.5) * valid).sum() / valid.sum().clamp_min(1.)


@MODELS.register_module()
class KDStudent(TopdownPoseEstimator):
    def loss(self, inputs, data_samples):
        if not self.training:
            return super().loss(inputs, data_samples)
        # Teacher lives outside the module tree: optimizer, EMA and state_dict
        # work exclusively on the original student parameters.
        if not hasattr(self, "_teacher"):
            raise RuntimeError("Attach frozen teacher before training")
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            target = self._teacher.head(self._teacher.extract_feat(inputs.float()))
        return self.head.loss(self.extract_feat(inputs), data_samples,
                              train_cfg=self.train_cfg, teacher_target=target)


@MODELS.register_module()
class KDHead(CIGHead):
    def loss(self, feats, batch_data_samples, train_cfg=None, teacher_target=None):
        gt_x = torch.cat([d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples], dim=0)
        gt_y = torch.cat([d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples], dim=0)
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples], dim=0)
        gt_simcc = (gt_x, gt_y)
        
        # --- Common feature extraction ---
        x = self.final_layer(feats[-1])
        x = torch.flatten(x, 2)
        f_kpts_initial = self.mlp_pre_gau(x)
        f_kpts = self.gau(f_kpts_initial)

        # --- Counterfactual Path (for the main loss L_kpt) ---
        h_initial_x = self.initial_pred_x(f_kpts.detach()) # Detach gradients from this branch, used only for CIM decision-making
        h_initial_y = self.initial_pred_y(f_kpts.detach())
        f_prime_kpts, intervened_mask = self.cim(f_kpts, (h_initial_x, h_initial_y))
        gcn_out_cf = self.gcn(f_prime_kpts)
        pred_x_cf = self.cls_gx(gcn_out_cf)
        pred_y_cf = self.cls_gy(gcn_out_cf)
        pred_simcc_cf = (pred_x_cf, pred_y_cf)

        # --- Observational Path (for the consistency loss L_cf) ---
        with torch.no_grad():
            gcn_out_obs = self.gcn(f_kpts)
            pred_x_obs = self.cls_gx(gcn_out_obs)
            pred_y_obs = self.cls_gy(gcn_out_obs)
            pred_simcc_obs = (pred_x_obs, pred_y_obs)

        # --- Loss Calculation ---
        losses = dict()
        loss_kpt = self.loss_module(pred_simcc_cf, gt_simcc, keypoint_weights)
        losses['loss_kpt'] = loss_kpt

        loss_cf = self.loss_cf_module(pred_simcc_obs, pred_simcc_cf, intervened_mask, keypoint_weights)
        losses['loss_cf'] = loss_cf
        
        if teacher_target is not None:
            raw_kd = simcc_kd(pred_simcc_cf, teacher_target, keypoint_weights)
            losses['kd_raw'] = raw_kd.detach()
            losses['loss_kd'] = train_cfg['kd_weight'] * raw_kd

        # --- Accuracy Calculation ---
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc_cf),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0)
        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses['acc_pose'] = acc_pose

        return losses
    

@METRICS.register_module()
class OccludedJointError(BaseMetric):
    default_prefix = "occluded"

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            ann = sample["raw_ann_info"]
            gt = np.asarray(ann["keypoints"]).reshape(17, 3)
            mask = gt[:, 2] == 1
            pred = np.asarray(sample["pred_instances"]["keypoints"]).reshape(17, 2)
            errors = np.linalg.norm(pred[mask] - gt[mask, :2], axis=1)
            self.results.append((len(errors), errors.sum(), errors.sum()/np.sqrt(ann["area"])))

    def compute_metrics(self, results):
        count, pixels, normalized = np.sum(results, axis=0)
        assert count > 0
        return dict(count=int(count), mean_px=float(pixels/count),
                    mean_area_normalized=float(normalized/count))
