"""
Unified AP/AR Evaluation Entry for CIGPose FYP (Role M).

Strictly aligned with docs/fyp/evaluation.md:
- Fixed protocol: COCO val2017 / GT boxes / single-view (flip_test=False)
- Supports BOTH input stages:
  1) pre_nms: instance-level predictions -> bbox_keypoint scoring (thr=0.2)
              + oks_nms (thr=0.9, area=bbox_scales[0]*bbox_scales[1]) -> COCOeval
  2) coco_json: already post-processed CocoMetric JSON (51-len keypoints + score)
                -> directly evaluated by COCOeval without re-scoring or re-NMS
- Never restricts COCOeval imgIds to only predicted images (evaluates full val2017)
"""

import argparse
import json
import os
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[3]

COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], dtype=np.float32)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    return np.stack([x, y, x + w, y + h], axis=-1)


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode('utf-8').strip()
    except Exception:
        return 'unknown'


def _compute_oks_matrix(
    kpts_a: np.ndarray,
    areas_a: np.ndarray,
    kpts_b: np.ndarray,
    areas_b: np.ndarray,
    sigmas: np.ndarray = COCO_SIGMAS,
) -> np.ndarray:
    vars_ = (sigmas * 2.0) ** 2
    dx = kpts_a[:, None, :, 0] - kpts_b[None, :, :, 0]
    dy = kpts_a[:, None, :, 1] - kpts_b[None, :, :, 1]
    d2 = dx ** 2 + dy ** 2
    area = np.maximum((areas_a[:, None] + areas_b[None, :]) / 2.0, 1.0)
    e = d2 / (2.0 * area[..., None] * vars_[None, None, :])
    return np.exp(-e).mean(axis=-1)


def _oks_nms(
    kpts: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    nms_thr: float = 0.9,
) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        oks = _compute_oks_matrix(
            kpts[i:i + 1], areas[i:i + 1],
            kpts[order[1:]], areas[order[1:]],
        )[0]
        inds = np.where(oks <= nms_thr)[0]
        order = order[inds + 1]
    return keep


class UnifiedPoseEvaluator:
    """Unified AP/AR evaluator aligned with docs/fyp/evaluation.md."""

    def __init__(
        self,
        ann_file: Union[str, Path],
        score_mode: str = 'bbox_keypoint',
        nms_mode: str = 'oks_nms',
        nms_thr: float = 0.9,
        keypoint_score_thr: float = 0.2,
    ) -> None:
        self.ann_file = str(ann_file)
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f'Annotation file not found: {self.ann_file}')

        self.coco_gt = COCO(self.ann_file)
        self.score_mode = score_mode
        self.nms_mode = nms_mode
        self.nms_thr = nms_thr
        self.keypoint_score_thr = keypoint_score_thr
        self._per_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._ann_ids_seen: List[int] = []

    def reset(self) -> None:
        self._per_image.clear()
        self._ann_ids_seen.clear()

    def add_batch(
        self,
        image_ids: Sequence[int],
        coords: Union[np.ndarray, torch.Tensor],
        scores: Union[np.ndarray, torch.Tensor],
        bboxes: Union[np.ndarray, torch.Tensor],
        bbox_scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
        bbox_scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
        annotation_ids: Optional[Sequence[int]] = None,
        bbox_format: str = 'xyxy',
    ) -> None:
        """Add pre-NMS instance predictions in original-image coordinates."""
        coords_np = _to_numpy(coords).astype(np.float32)
        scores_np = _to_numpy(scores).astype(np.float32)
        bboxes_np = _to_numpy(bboxes).astype(np.float32)

        if np.isnan(coords_np).any() or np.isinf(coords_np).any():
            raise ValueError('Predicted coords contain NaN or Inf.')

        if bbox_format == 'xywh':
            bboxes_np = _xywh_to_xyxy(bboxes_np)
        elif bbox_format != 'xyxy':
            raise ValueError(f'Unsupported bbox_format: {bbox_format}')

        bsz = len(image_ids)
        if bbox_scores is None:
            bbox_scores_np = np.ones((bsz,), dtype=np.float32)
        else:
            bbox_scores_np = _to_numpy(bbox_scores).astype(np.float32)

        # Follow evaluation.md: NMS area uses bbox_scales[0] * bbox_scales[1]
        if bbox_scales is not None:
            scales_np = _to_numpy(bbox_scales).astype(np.float32)
            areas_np = scales_np[:, 0] * scales_np[:, 1]
        else:
            areas_np = np.maximum(0.0, bboxes_np[:, 2] - bboxes_np[:, 0]) * \
                np.maximum(0.0, bboxes_np[:, 3] - bboxes_np[:, 1])

        for i in range(bsz):
            kpt_s = scores_np[i]
            valid = kpt_s > self.keypoint_score_thr
            mean_kpt_score = float(kpt_s[valid].mean()) if valid.any() else 0.0

            if self.score_mode == 'bbox_keypoint':
                pose_score = float(bbox_scores_np[i]) * mean_kpt_score
            elif self.score_mode == 'keypoint':
                pose_score = mean_kpt_score
            else:
                pose_score = float(bbox_scores_np[i])

            if annotation_ids is not None:
                self._ann_ids_seen.append(int(annotation_ids[i]))

            self._per_image[int(image_ids[i])].append({
                'coords': coords_np[i],
                'scores': kpt_s,
                'area': float(areas_np[i]),
                'pose_score': pose_score,
            })

    def add_records(self, records: Sequence[Dict[str, Any]]) -> None:
        """Add list of pre-NMS dict records matching evaluation.md / interfaces.md."""
        if not records:
            return
        image_ids = [int(r['image_id']) for r in records]
        coords_key = 'keypoints' if 'keypoints' in records[0] else 'coords'
        scores_key = 'keypoint_scores' if 'keypoint_scores' in records[0] else 'scores'
        coords = np.stack([_to_numpy(r[coords_key]) for r in records], axis=0)
        scores = np.stack([_to_numpy(r[scores_key]) for r in records], axis=0)
        bboxes = np.stack([_to_numpy(r['bbox']) for r in records], axis=0)
        bbox_scores = np.array(
            [float(r.get('bbox_score', 1.0)) for r in records],
            dtype=np.float32,
        )
        scale_key = 'bbox_scales' if 'bbox_scales' in records[0] else (
            'input_scale' if 'input_scale' in records[0] else None
        )
        bbox_scales = (
            np.stack([_to_numpy(r[scale_key]) for r in records], axis=0)
            if scale_key else None
        )
        ann_ids = (
            [int(r['annotation_id']) for r in records]
            if 'annotation_id' in records[0] else None
        )
        bbox_format = records[0].get('bbox_format', 'xyxy')
        self.add_batch(
            image_ids=image_ids,
            coords=coords,
            scores=scores,
            bboxes=bboxes,
            bbox_scores=bbox_scores,
            bbox_scales=bbox_scales,
            annotation_ids=ann_ids,
            bbox_format=bbox_format,
        )

    def evaluate_coco_results(
        self, coco_results: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Directly evaluate already post-processed COCO results (skip scoring & NMS)."""
        if not coco_results:
            raise RuntimeError('Empty COCO results list.')

        coco_dt = self.coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stat_names = [
            'coco/AP', 'coco/AP .5', 'coco/AP .75', 'coco/AP (M)', 'coco/AP (L)',
            'coco/AR', 'coco/AR .5', 'coco/AR .75', 'coco/AR (M)', 'coco/AR (L)',
        ]
        metrics = {name: float(coco_eval.stats[i]) for i, name in enumerate(stat_names)}
        audit = {
            'num_output_predictions': len(coco_results),
            'num_unique_images': len({int(r['image_id']) for r in coco_results}),
        }
        return metrics, audit

    def evaluate(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Apply scoring + OKS NMS on pre-NMS predictions, then run COCOeval."""
        coco_results: List[Dict[str, Any]] = []
        total_before_nms = 0

        for img_id, items in self._per_image.items():
            total_before_nms += len(items)
            if self.nms_mode == 'oks_nms' and len(items) > 1:
                kpts = np.stack([x['coords'] for x in items], axis=0)
                pose_scores = np.array([x['pose_score'] for x in items], dtype=np.float32)
                areas = np.array([x['area'] for x in items], dtype=np.float32)
                keep = _oks_nms(kpts, pose_scores, areas, nms_thr=self.nms_thr)
            else:
                keep = list(range(len(items)))

            for idx in keep:
                it = items[idx]
                kpts_3 = np.concatenate(
                    [it['coords'], it['scores'][:, None]], axis=-1
                )
                coco_results.append({
                    'image_id': img_id,
                    'category_id': 1,
                    'keypoints': kpts_3.reshape(-1).tolist(),
                    'score': float(it['pose_score']),
                })

        if not coco_results:
            raise RuntimeError('No predictions were added to UnifiedPoseEvaluator.')

        metrics, audit = self.evaluate_coco_results(coco_results)
        audit['num_input_instances'] = total_before_nms
        if self._ann_ids_seen:
            unique_anns = len(set(self._ann_ids_seen))
            audit['num_annotation_ids'] = len(self._ann_ids_seen)
            audit['num_unique_annotation_ids'] = unique_anns
            audit['duplicate_annotation_ids'] = len(self._ann_ids_seen) - unique_anns
        return metrics, audit


def save_eval_report(
    metrics: Dict[str, float],
    audit: Dict[str, Any],
    output_path: Union[str, Path],
    exp_name: str,
    stage: str,
    ann_file: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'exp_name': exp_name,
        'protocol_label': 'COCO val2017 / GT boxes / single-view',
        'input_stage': stage,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'git_commit': _get_git_commit(),
        'flip_protocol': 'single_view',
        'flip_test': False,
        'bbox_source': 'gt_bbox',
        'score_mode': 'bbox_keypoint',
        'keypoint_score_thr': 0.2,
        'nms_mode': 'oks_nms',
        'nms_thr': 0.9,
        'ann_file': str(ann_file),
        'audit': audit,
        'metrics': metrics,
    }
    if extra_meta:
        payload['meta'] = extra_meta

    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


def _load_predictions(pred_path: Path) -> Any:
    suffix = pred_path.suffix.lower()
    if suffix in ('.pt', '.pth'):
        return torch.load(pred_path, map_location='cpu')
    if suffix == '.pkl':
        with open(pred_path, 'rb') as f:
            return pickle.load(f)
    if suffix == '.json':
        with open(pred_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise ValueError(f'Unsupported prediction file format: {suffix}')


def _detect_stage(preds: Any) -> str:
    if isinstance(preds, list) and preds and isinstance(preds[0], dict):
        kpts = preds[0].get('keypoints')
        if isinstance(kpts, list) and len(kpts) == 51 and 'score' in preds[0]:
            return 'coco_json'
    return 'pre_nms'


def _run_smoke_test(ann_file: str) -> None:
    print(f'Running smoke test (pre_nms + coco_json) on: {ann_file}')
    evaluator = UnifiedPoseEvaluator(ann_file=ann_file)
    ann_ids = evaluator.coco_gt.getAnnIds(iscrowd=False)
    anns = evaluator.coco_gt.loadAnns(ann_ids)

    image_ids, coords, scores, bboxes, scales, valid_ann_ids = [], [], [], [], [], []
    for ann in anns:
        kpts = np.array(ann['keypoints'], dtype=np.float32).reshape(17, 3)
        if np.sum(kpts[:, 2] > 0) == 0:
            continue
        image_ids.append(int(ann['image_id']))
        valid_ann_ids.append(int(ann['id']))
        coords.append(kpts[:, :2])
        scores.append(np.where(kpts[:, 2] > 0, 1.0, 0.0).astype(np.float32))
        bboxes.append(ann['bbox'])
        scales.append([max(ann['bbox'][2], 1.0) * 1.25, max(ann['bbox'][3], 1.0) * 1.25])

    # 1) Test pre_nms path
    evaluator.add_batch(
        image_ids=image_ids,
        coords=np.stack(coords, axis=0),
        scores=np.stack(scores, axis=0),
        bboxes=np.stack(bboxes, axis=0),
        bbox_scales=np.stack(scales, axis=0),
        annotation_ids=valid_ann_ids,
        bbox_format='xywh',
    )
    metrics_pre, audit_pre = evaluator.evaluate()
    print(f'\n[Smoke Test - pre_nms] audit={audit_pre}')
    print(f'[Smoke Test - pre_nms] coco/AP={metrics_pre["coco/AP"]:.4f}, coco/AR={metrics_pre["coco/AR"]:.4f}')

    # 2) Test coco_json path (no re-NMS)
    coco_results = []
    for img_id, c, s in zip(image_ids, coords, scores):
        kpts_3 = np.concatenate([c, s[:, None]], axis=-1)
        coco_results.append({
            'image_id': img_id,
            'category_id': 1,
            'keypoints': kpts_3.reshape(-1).tolist(),
            'score': 1.0,
        })
    metrics_json, audit_json = evaluator.evaluate_coco_results(coco_results)
    print(f'\n[Smoke Test - coco_json (no NMS)] audit={audit_json}')
    print(f'[Smoke Test - coco_json (no NMS)] coco/AP={metrics_json["coco/AP"]:.4f}, coco/AR={metrics_json["coco/AR"]:.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified AP/AR Evaluator (Role M)')
    parser.add_argument('--ann-file', type=str, required=True)
    parser.add_argument('--pred-file', type=str, default=None)
    parser.add_argument(
        '--stage',
        type=str,
        default='auto',
        choices=['auto', 'pre_nms', 'coco_json'],
        help='Prediction stage: pre_nms (needs scoring+NMS) or coco_json (already post-processed)',
    )
    parser.add_argument('--exp-name', type=str, default='eval_run')
    parser.add_argument('--out-json', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(args.ann_file)
        return

    if not args.pred_file:
        raise ValueError('Either --pred-file or --smoke-test must be provided.')

    evaluator = UnifiedPoseEvaluator(ann_file=args.ann_file)
    preds = _load_predictions(Path(args.pred_file))
    stage = _detect_stage(preds) if args.stage == 'auto' else args.stage

    if stage == 'coco_json':
        metrics, audit = evaluator.evaluate_coco_results(preds)
    elif isinstance(preds, dict):
        coords = preds.get('keypoints', preds.get('coords'))
        scores = preds.get('keypoint_scores', preds.get('scores'))
        scales = preds.get('bbox_scales', preds.get('input_scale'))
        evaluator.add_batch(
            image_ids=preds['image_id'],
            coords=coords,
            scores=scores,
            bboxes=preds['bbox'],
            bbox_scores=preds.get('bbox_score'),
            bbox_scales=scales,
            annotation_ids=preds.get('annotation_id'),
            bbox_format=preds.get('bbox_format', 'xyxy'),
        )
        metrics, audit = evaluator.evaluate()
    elif isinstance(preds, list):
        evaluator.add_records(preds)
        metrics, audit = evaluator.evaluate()
    else:
        raise TypeError(f'Unexpected prediction container type: {type(preds)}')

    print(f'\n=== Unified Evaluation Results ({args.exp_name} | stage={stage}) ===')
    print(f'Audit: {audit}')
    for k, v in metrics.items():
        print(f'  {k}: {v:.5f}')

    out_path = args.out_json or (
        REPO_ROOT / 'experiments' / 'baseline' / f'{args.exp_name}_eval.json'
    )
    saved = save_eval_report(
        metrics=metrics,
        audit=audit,
        output_path=out_path,
        exp_name=args.exp_name,
        stage=stage,
        ann_file=args.ann_file,
        extra_meta={'pred_file': args.pred_file},
    )
    print(f'\nSaved report to: {saved}')


if __name__ == '__main__':
    main()