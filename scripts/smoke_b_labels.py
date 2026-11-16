"""B-only smoke labels: fixed uncertainty Top-1, four non-zero strengths.

Run from any directory with --repo pointing to the shared project checkout.
This is a bounded first-shard check, NOT a full-label generator or AP evaluation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

# Set before importing torch or initializing CUDA.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

STRENGTHS = (0.25, 0.5, 0.75, 1.0)
WEIGHT_SHA = '3dd69c8ab4e3c7a923a0183fd9abf844cb7ef13d39d26693baa082ca7931b3d0'
TRAIN_SHA = '7fc1549d934547c470384d8a207c38707ca33fc016d00e33b795e408603af83e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--ann', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=32, choices=range(1, 33))
    args = p.parse_args()
    for key in ('repo', 'cache', 'ann', 'checkpoint', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists():
        raise FileExistsError('Use a NEW output directory: ' + str(args.output))
    if sha(args.checkpoint) != WEIGHT_SHA or sha(args.ann) != TRAIN_SHA:
        raise ValueError('Checkpoint or original train annotation SHA mismatch')
    mmroot = args.repo / 'mmpose'
    sys.path[:0] = [str(mmroot), str(mmroot / 'analysis/fyp')]
    os.chdir(mmroot)

    import numpy as np
    import torch
    from mmengine.config import Config
    from mmpose.apis import init_model
    from shared_cache import ReferenceCache
    from build_shared_cache import candidate_from_cache
    from oks_utility import compute_instance_oks, compute_gain

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError('This smoke run requires the configured CUDA environment')

    cache = ReferenceCache(args.cache)
    if cache.run['protocol'].get('dataset_split') != 'train2017':
        raise ValueError('Use the train2017 GT cache, not val/detection data')
    protocol = cache.run['protocol']
    if (protocol.get('checkpoint_sha256') != WEIGHT_SHA
            or protocol.get('annotation_sha256') != TRAIN_SHA
            or protocol.get('bbox_source') != 'gt'
            or protocol.get('flip_test') is not False):
        raise ValueError('Cache provenance differs from the shared train protocol')
    # The full cache has already passed the delivered read-only validator.
    # Validate the selected shard with the public reader; do not rescan 4,682 shards.
    first_shard = cache.shards()[0]
    full_batch = cache.load_shard(first_shard)
    if len(full_batch['annotation_id']) < args.count:
        raise ValueError('First shard contains fewer than requested samples')
    batch = {k: v[:args.count].copy() for k, v in full_batch.items()}
    n = len(batch['annotation_id'])
    coco = json.loads(args.ann.read_text(encoding='utf-8-sig'))
    anns = {int(a['id']): a for a in coco['annotations']}
    gt, areas = [], []
    for img_id, ann_id in zip(batch['image_id'], batch['annotation_id']):
        a = anns[int(ann_id)]
        if (int(a['image_id']) != int(img_id) or a['category_id'] != 1
                or a.get('iscrowd', 0) != 0):
            raise ValueError('GT identity/category/crowd mismatch')
        gt.append(np.asarray(a['keypoints'], dtype=np.float64).reshape(17, 3))
        areas.append(float(a['area']))
    del coco, anns
    gt, areas = np.asarray(gt), np.asarray(areas)
    oks0 = np.asarray([compute_instance_oks(batch['P_0_coords'][i], gt[i], areas[i])
                       for i in range(n)])
    original = {k: v.copy() for k, v in batch.items()}
    cfg_path = mmroot / ('projects/cigpose/body_2d_keypoint/'
                         'cigpose-m_1xb384-420e_coco-256x192.py')
    cfg = Config.fromfile(str(cfg_path))
    cfg.model.test_cfg['flip_test'] = False
    model = init_model(cfg, str(args.checkpoint), device='cuda:0')
    model.requires_grad_(False).eval()
    joints = torch.as_tensor(batch['top1_joint'], dtype=torch.long, device='cuda:0')
    if not np.array_equal(batch['top1_joint'], batch['uncertainty'].argmax(axis=1)):
        raise ValueError('Cached Top-1 differs from uncertainty argmax')

    def candidate(alpha):
        with torch.inference_mode():
            result = candidate_from_cache(model.head, batch, joints,
                torch.full((n,), alpha, dtype=torch.float32, device='cuda:0'), 'cuda:0')
        for k in batch:
            np.testing.assert_array_equal(batch[k], original[k])
        return result

    def compare(c, s, ref_c, ref_s):
        if not np.isfinite(c).all() or not np.isfinite(s).all():
            raise ValueError('Non-finite candidate')
        np.testing.assert_allclose(c, ref_c, atol=0.1, rtol=0)
        np.testing.assert_allclose(s, ref_s, atol=1e-5, rtol=1e-6)
        np.testing.assert_array_equal(s > 0.2, ref_s > 0.2)
        return {'coordinate_max_abs_error': float(np.abs(c-ref_c).max()),
                'score_max_abs_error': float(np.abs(s-ref_s).max())}

    c0, s0 = candidate(0.0)
    alpha0_check = compare(c0, s0, batch['P_0_coords'], batch['P_0_scores'])
    alpha0_check['max_abs_gain'] = max(abs(compute_gain(c0[i],
        batch['P_0_coords'][i], gt[i], areas[i])) for i in range(n))
    outputs = {}
    for alpha in STRENGTHS:
        outputs[alpha] = candidate(alpha)
        print('Generated alpha=' + str(alpha), flush=True)
    # Reorder calls to detect shared-state/order dependence at public tolerances.
    order_checks = {}
    for alpha in reversed(STRENGTHS):
        c, s = candidate(alpha)
        order_checks[str(alpha)] = compare(c, s, *outputs[alpha])

    rows, summaries = [], {}
    gains = np.empty((n, 4), dtype=np.float64)
    for j, alpha in enumerate(STRENGTHS):
        coords, scores = outputs[alpha]
        for i in range(n):
            after = compute_instance_oks(coords[i], gt[i], areas[i])
            gain = compute_gain(coords[i], batch['P_0_coords'][i], gt[i], areas[i])
            np.testing.assert_allclose(gain, after-oks0[i], atol=1e-12, rtol=0)
            gains[i, j] = gain
            rows.append(dict(dataset_split='train2017', image_id=int(batch['image_id'][i]),
                annotation_id=int(batch['annotation_id'][i]),
                joint_id=int(batch['top1_joint'][i]), alpha=alpha,
                gt_keypoints=gt[i].tolist(), gt_area=float(areas[i]),
                oks_before=float(oks0[i]), oks_after=after, gain=gain))
        g = gains[:, j]
        summaries[str(alpha)] = dict(mean_gain=float(g.mean()), min_gain=float(g.min()),
            max_gain=float(g.max()), positive=int((g > 0).sum()),
            negative=int((g < 0).sum()), zero=int((g == 0).sum()))
    best = gains.argmax(axis=1)  # Exact ties choose first (smallest alpha).
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    meta = dict(schema_version='b-only-smoke-labels-v1', smoke_only=True,
        sampling='first N records of first train GT shard; not representative',
        instances=n, candidate_records=len(rows), selection='uncertainty_top1',
        alphas=list(STRENGTHS), always_accept=True, alpha_zero_is_class=False,
        git_commit=commit, checkpoint_sha256=WEIGHT_SHA, annotation_sha256=TRAIN_SHA,
        cache_run_sha256=sha(args.cache/'run.json'), source_shard=str(first_shard),
        source_shard_sha256=sha(first_shard), cache_protocol=cache.run['protocol'],
        current_runtime=dict(seed=42, deterministic_algorithms=True,
            CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False,
            cudnn_deterministic=True, precision='float32', flip_test=False,
            base_frozen=True, base_eval=True, device=torch.cuda.get_device_name(0)),
        versions={k: importlib.metadata.version(k) for k in
                  ['torch', 'torchvision', 'numpy', 'mmcv', 'mmengine', 'mmpose']},
        code_sha256={str(f): sha(f) for f in [Path(__file__).resolve(), cfg_path,
            mmroot/'analysis/fyp/build_shared_cache.py', mmroot/'analysis/fyp/oks_utility.py',
            mmroot/'mmpose/models/heads/coord_cls_heads/cig_head.py']},
        alpha0=alpha0_check, candidate_order_checks=order_checks,
        source_arrays_unchanged=True, gain_summary=summaries,
        oracle_best_alpha_counts={str(a): int((best == j).sum())
                                  for j, a in enumerate(STRENGTHS)},
        oracle_note='Offline analysis only. Exact ties choose smallest alpha; all four gains retained.',
        limitations=['No training, no formal AP/AR, no full label coverage.',
            'Candidate generation and decoding reuse the public helper.',
            'Order checks do not independently prove alpha=1 replacement mathematics.',
            'Old cache metadata preserved; current deterministic settings recorded separately.'])
    # Publish only after all computations/checks pass; never overwrite old runs.
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'candidates.npz',
        annotation_id=batch['annotation_id'], image_id=batch['image_id'],
        joint_id=batch['top1_joint'], alphas=np.asarray(STRENGTHS, dtype=np.float32),
        P_0_coords=batch['P_0_coords'], P_0_scores=batch['P_0_scores'],
        candidate_coords=np.stack([outputs[a][0] for a in STRENGTHS], axis=1),
        candidate_scores=np.stack([outputs[a][1] for a in STRENGTHS], axis=1),
        gain=gains, best_alpha_index=best)
    with np.load(args.output/'candidates.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['gain'], gains)
        np.testing.assert_array_equal(saved['annotation_id'], batch['annotation_id'])
        np.testing.assert_array_equal(saved['candidate_coords'],
            np.stack([outputs[a][0] for a in STRENGTHS], axis=1))
    write_json(args.output/'labels.json', rows)
    write_json(args.output/'report.json', meta)
    cfg.dump(str(args.output/'resolved_config.py'))
    write_json(args.output/'SHA256.json', {f.name: sha(f) for f in sorted(args.output.iterdir())})
    print(json.dumps({'PASS': True, 'instances': n, 'candidate_records': len(rows),
                      'output': str(args.output), 'gain_summary': summaries}, indent=2))


if __name__ == '__main__':
    main()
