"""
Module C: Candidate Acceptance, Label Audit, Threshold Sweep & Fallback (Role M + T).

Aligned with Proposal Section 4.1.3 & 4.3, docs/fyp/interfaces.md, and public-tasks:
- Strict 1-to-1 pairing between P_0 and P_c pre-NMS predictions
- Offline GT matching (for supervision & error analysis ONLY; never deletes FP detections)
- Uses LIN's oks_utility.compute_gain for binary acceptance labels (gain > 0 -> 1, else 0)
- Scale-normalized pose-change feature encoder & lightweight MLP classifier
- Threshold-based fallback (p_acc > tau -> P_c, else P_0) & UnifiedPoseEvaluator AP/AR
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO

FYP_DIR = Path(__file__).resolve().parent
REPO_ROOT = FYP_DIR.parents[2]
if str(FYP_DIR) not in sys.path:
    sys.path.insert(0, str(FYP_DIR))

from eval_unified import UnifiedPoseEvaluator, save_eval_report  # noqa: E402
from oks_utility import compute_gain  # noqa: E402


def _xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    return np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]], dtype=np.float64)


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def pair_p0_and_pc(
    p0_records: Sequence[Dict[str, Any]],
    pc_records: Sequence[Dict[str, Any]],
) -> None:
    """Verify strict 1-to-1 instance alignment between P_0 and P_c predictions."""
    if len(p0_records) != len(pc_records):
        raise ValueError(
            f'Record count mismatch: P_0={len(p0_records)} vs P_c={len(pc_records)}'
        )
    for idx, (r0, rc) in enumerate(zip(p0_records, pc_records)):
        if int(r0['image_id']) != int(rc['image_id']):
            raise ValueError(f'image_id mismatch at index {idx}')
        if r0.get('detection_instance_id') != rc.get('detection_instance_id'):
            raise ValueError(f'detection_instance_id mismatch at index {idx}')
        if not np.allclose(r0['bbox'], rc['bbox'], atol=1e-4):
            raise ValueError(f'bbox mismatch at index {idx}')


def match_detections_to_gt_and_compute_gains(
    p0_records: Sequence[Dict[str, Any]],
    pc_records: Sequence[Dict[str, Any]],
    ann_file: str,
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """Offline GT matching & OKS gain labeling (for supervision/analysis ONLY)."""
    coco_gt = COCO(ann_file)

    # Group detection indices by image_id
    img_to_det_indices: Dict[int, List[int]] = {}
    for idx, r in enumerate(p0_records):
        img_to_det_indices.setdefault(int(r['image_id']), []).append(idx)

    gains = np.full(len(p0_records), np.nan, dtype=np.float64)
    labels = np.full(len(p0_records), -1, dtype=np.int64)
    matched_iou = np.zeros(len(p0_records), dtype=np.float64)

    for img_id, det_indices in img_to_det_indices.items():
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco_gt.loadAnns(ann_ids)

        # Keep valid GT persons (labeled keypoints > 0 and positive area)
        valid_gts = []
        for ann in anns:
            kpts = np.asarray(ann['keypoints'], dtype=np.float64).reshape(17, 3)
            area = float(ann.get('area', 0.0))
            if np.any(kpts[:, 2] > 0) and area > 0.0:
                x, y, w, h = ann['bbox']
                valid_gts.append({
                    'kpts': kpts,
                    'area': area,
                    'bbox_xyxy': np.array([x, y, x + w, y + h], dtype=np.float64),
                })

        if not valid_gts:
            continue

        # Sort detections by bbox_score descending for standard greedy matching
        sorted_dets = sorted(
            det_indices,
            key=lambda i: float(p0_records[i].get('bbox_score', 1.0)),
            reverse=True,
        )
        used_gt = set()
        for det_i in sorted_dets:
            det_box = np.asarray(p0_records[det_i]['bbox'], dtype=np.float64)
            best_gt = -1
            best_iou = iou_thr
            for gt_j, gt in enumerate(valid_gts):
                if gt_j in used_gt:
                    continue
                iou = _box_iou_xyxy(det_box, gt['bbox_xyxy'])
                if iou >= best_iou:
                    best_iou = iou
                    best_gt = gt_j
            if best_gt >= 0:
                used_gt.add(best_gt)
                gt = valid_gts[best_gt]
                p0_kpts = np.asarray(p0_records[det_i]['keypoints'], dtype=np.float64)
                pc_kpts = np.asarray(pc_records[det_i]['keypoints'], dtype=np.float64)
                g = compute_gain(pc_kpts, p0_kpts, gt['kpts'], gt['area'])
                gains[det_i] = g
                labels[det_i] = 1 if g > 0.0 else 0
                matched_iou[det_i] = best_iou

    matched_mask = labels >= 0
    matched_gains = gains[matched_mask]
    pos_count = int(np.sum(matched_gains > 0.0))
    zero_count = int(np.sum(matched_gains == 0.0))
    neg_count = int(np.sum(matched_gains < 0.0))
    total_matched = int(np.sum(matched_mask))

    stats = {
        'total_detections': len(p0_records),
        'matched_gt_detections': total_matched,
        'unmatched_detections': len(p0_records) - total_matched,
        'positive_gain_count': pos_count,
        'zero_gain_count': zero_count,
        'negative_gain_count': neg_count,
        'positive_ratio': pos_count / max(total_matched, 1),
        'zero_ratio': zero_count / max(total_matched, 1),
        'negative_ratio': neg_count / max(total_matched, 1),
        'mean_gain_matched': float(np.mean(matched_gains)) if total_matched > 0 else 0.0,
    }
    return {
        'gains': gains,
        'labels': labels,
        'matched_mask': matched_mask,
        'stats': stats,
    }


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
    """Encode scale-normalized pose change & intervention metadata for Module C.

    Args:
        p0_coords: [B, 17, 2]
        pc_coords: [B, 17, 2]
        bboxes_xyxy: [B, 4]
        p0_scores: [B, 17]
        pc_scores: [B, 17]
        joint_ids: [B] (0..16)
        alphas: [B]
        uncertainty: optional [B, 17]
    Returns:
        Feature tensor [B, D] (D=120 with uncertainty, D=103 without)
    """
    cx = (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) * 0.5
    cy = (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) * 0.5
    center = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # [B, 1, 2]
    w = torch.clamp(bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0], min=1.0)
    h = torch.clamp(bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1], min=1.0)
    scale = torch.maximum(w, h).view(-1, 1, 1)  # [B, 1, 1]

    p0_norm = ((p0_coords - center) / scale).reshape(p0_coords.size(0), -1)        # [B, 34]
    delta_norm = ((pc_coords - p0_coords) / scale).reshape(p0_coords.size(0), -1)  # [B, 34]
    delta_scores = pc_scores - p0_scores                                            # [B, 17]
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


def apply_acceptance_fallback(
    p0_records: Sequence[Dict[str, Any]],
    pc_records: Sequence[Dict[str, Any]],
    accept_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """Select P_c where accept_mask is True, else safely fall back to P_0."""
    out: List[Dict[str, Any]] = []
    for i, (r0, rc) in enumerate(zip(p0_records, pc_records)):
        chosen = rc if bool(accept_mask[i]) else r0
        rec = dict(r0)
        rec['keypoints'] = chosen['keypoints']
        rec['keypoint_scores'] = chosen['keypoint_scores']
        out.append(rec)
    return out


def compute_decision_error_analysis(
    gains: np.ndarray,
    matched_mask: np.ndarray,
    accept_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute Proposal 4.3 metrics on GT-matched detections."""
    g = gains[matched_mask]
    acc = accept_mask[matched_mask]

    neg = g < 0.0
    non_pos = g <= 0.0
    pos = g > 0.0

    harmful_elim_rate = float(np.sum((~acc) & neg) / max(np.sum(neg), 1))
    false_acc_rate = float(np.sum(acc & non_pos) / max(np.sum(non_pos), 1))
    false_rej_rate = float(np.sum((~acc) & pos) / max(np.sum(pos), 1))

    realized_gain = np.where(acc, g, 0.0)
    return {
        'accept_rate_all': float(np.mean(accept_mask)),
        'accept_rate_matched': float(np.mean(acc)),
        'harmful_elimination_rate': harmful_elim_rate,
        'false_acceptance_rate': false_acc_rate,
        'false_rejection_rate': false_rej_rate,
        'mean_realized_gain': float(np.mean(realized_gain)),
        'mean_gain_accepted': float(np.mean(g[acc])) if np.any(acc) else 0.0,
        'mean_gain_rejected_if_applied': float(np.mean(g[~acc])) if np.any(~acc) else 0.0,
    }


def evaluate_acceptance_policy(
    p0_records: Sequence[Dict[str, Any]],
    pc_records: Sequence[Dict[str, Any]],
    accept_mask: np.ndarray,
    ann_file: str,
    gains: Optional[np.ndarray] = None,
    matched_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run full-104k UnifiedPoseEvaluator + decision error analysis."""
    merged = apply_acceptance_fallback(p0_records, pc_records, accept_mask)
    evaluator = UnifiedPoseEvaluator(ann_file=ann_file)
    evaluator.add_records(merged)
    metrics, audit = evaluator.evaluate()

    result: Dict[str, Any] = {'metrics': metrics, 'audit': audit}
    if gains is not None and matched_mask is not None:
        result['decision_analysis'] = compute_decision_error_analysis(
            gains=gains, matched_mask=matched_mask, accept_mask=accept_mask
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Module C Audit, Oracle & Threshold Analysis')
    parser.add_argument('--p0-file', type=str, required=True)
    parser.add_argument('--pc-file', type=str, required=True)
    parser.add_argument('--ann-file', type=str, required=True)
    parser.add_argument('--out-json', type=str, default=None)
    parser.add_argument('--run-oracle', action='store_true', help='Evaluate Oracle acceptance upper bound')
    args = parser.parse_args()

    print('Loading P_0 and P_c pre-NMS predictions...')
    with open(args.p0_file, 'r', encoding='utf-8') as f:
        p0_records = json.load(f)
    with open(args.pc_file, 'r', encoding='utf-8') as f:
        pc_records = json.load(f)

    pair_p0_and_pc(p0_records, pc_records)
    print(f'Pairing check PASSED: {len(p0_records)} detections aligned 1-to-1.')

    audit_res = match_detections_to_gt_and_compute_gains(
        p0_records=p0_records,
        pc_records=pc_records,
        ann_file=args.ann_file,
    )
    print('\n=== Module C Label Audit (Fixed Top-1 + alpha=1) ===')
    for k, v in audit_res['stats'].items():
        print(f'  {k}: {v:.5f}' if isinstance(v, float) else f'  {k}: {v}')

    payload: Dict[str, Any] = {'label_audit': audit_res['stats']}

    if args.run_oracle:
        # Oracle: accept iff matched and gain > 0; otherwise fall back to P_0
        oracle_accept = audit_res['matched_mask'] & (audit_res['gains'] > 0.0)
        print('\nRunning UnifiedPoseEvaluator for Oracle Acceptance (all 104,125 detections)...')
        oracle_eval = evaluate_acceptance_policy(
            p0_records=p0_records,
            pc_records=pc_records,
            accept_mask=oracle_accept,
            ann_file=args.ann_file,
            gains=audit_res['gains'],
            matched_mask=audit_res['matched_mask'],
        )
        payload['oracle_c_only'] = oracle_eval
        print('\n=== Oracle C-only Results ===')
        print(f"  coco/AP: {oracle_eval['metrics']['coco/AP']:.6f}")
        print(f"  coco/AR: {oracle_eval['metrics']['coco/AR']:.6f}")
        print(f"  decision_analysis: {oracle_eval['decision_analysis']}")

    out_path = Path(args.out_json) if args.out_json else (
        REPO_ROOT / 'experiments' / 'module_c' / 'c_only_audit_and_oracle.json'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'\nSaved Module C audit report to: {out_path}')


if __name__ == '__main__':
    main()