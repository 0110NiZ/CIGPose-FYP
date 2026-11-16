"""Generate the four B gain labels with the final near-tie audit."""
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib.metadata
import json
import random
import subprocess
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from Module_B.config import STRENGTHS, WEIGHT_SHA
from Module_B.common import sha, digest

def classify_near_ties(np, coords, old_coords, arrays, raw_logits, ratio):
    """Explain >0.1px differences only by near-equal raw SimCC bins.

    This is a diagnostic exception, not proof of historical logit equality.
    Historical logits were not saved. No coordinates/logits are changed.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('Invalid SimCC split ratio')
    bad = np.argwhere(np.abs(coords-old_coords) > .1)
    details = []
    for person, joint, axis in bad:
        center = float(arrays['input_center'][person, axis])
        scale = float(arrays['input_scale'][person, axis])
        size = float(arrays['input_size'][person, axis])
        def bin_position(value):
            return (float(value)-center+.5*scale)/scale*size*ratio
        cur_float = bin_position(coords[person, joint, axis])
        old_float = bin_position(old_coords[person, joint, axis])
        cur_bin, old_bin = int(round(cur_float)), int(round(old_float))
        values = raw_logits[axis][person, joint]
        if (values.dtype != np.float32 or not np.isfinite(values).all()
                or max(abs(cur_float-cur_bin), abs(old_float-old_bin)) > 1e-3
                or not (0 <= old_bin < len(values) and 0 <= cur_bin < len(values))
                or cur_bin != int(np.argmax(values))):
            raise ValueError('Unexplained old-cache coordinate mismatch: '
                             + str((int(person), int(joint), int(axis))))
        gap = float(values[cur_bin])-float(values[old_bin])
        bound = 8 * float(np.finfo(np.float32).eps) * max(
            1., abs(float(values[cur_bin])), abs(float(values[old_bin])))
        if not 0 <= gap <= bound:
            raise ValueError('Coordinate mismatch is not a near tie: '
                             + str((int(person), int(joint), int(axis), gap, bound)))
        details.append(dict(annotation_id=int(arrays['annotation_id'][person]),
            joint=int(joint), axis=int(axis), cached_bin=old_bin,
            current_bin=cur_bin, bin_distance=abs(cur_bin-old_bin), raw_logit_gap=gap, diagnostic_gap_bound=bound,
            cached_coordinate=float(old_coords[person,joint,axis]),
            current_coordinate=float(coords[person,joint,axis])))
    return details


def verify_near_ties(np, torch, model, arrays, coords, scores):
    # DARK/refined decoding does not satisfy the discrete-bin explanation.
    if getattr(model.head.decoder, 'use_dark', True):
        raise ValueError('Near-tie exception requires explicit use_dark=False')
    from build_shared_cache import array
    from shared_cache import restore_coords
    with torch.inference_mode():
        features = torch.as_tensor(arrays['features'], device='cuda:0')
        logits = model.head.predict_from_joint_features(features)
        raw = tuple(x.detach().cpu().numpy().copy() for x in logits)
        decoded = model.head.decode(tuple(x.clone() for x in logits))
    direct_coords = np.asarray([restore_coords(array(p.keypoints)[0,:,:2],
        arrays['input_center'][i], arrays['input_scale'][i], arrays['input_size'][i])
        for i, p in enumerate(decoded)])
    direct_scores = np.asarray([array(p.keypoint_scores)[0] for p in decoded])
    # The independent bypass must exactly reproduce the current alpha-zero path.
    np.testing.assert_array_equal(direct_coords, coords)
    np.testing.assert_array_equal(direct_scores, scores)
    return classify_near_ties(np, coords, arrays['P_0_coords'], arrays, raw,
                                 float(model.head.decoder.simcc_split_ratio))


def audit_legacy_reference(np, coords, scores, old_coords, old_scores, annotation_ids, near_tie_check):
    """Keep the original score acceptance result explicit, separate from supervision.

    OKS consumes coordinates only. The current no-intervention pose is used for
    labels; historical scores and threshold changes remain independently audited.
    """
    for x in (coords, scores, old_coords, old_scores):
        if not np.isfinite(x).all():
            raise ValueError('Non-finite current/historical reference')
    coordinate_passed = bool((np.abs(coords-old_coords) <= .1).all())
    near_ties = [] if coordinate_passed else near_tie_check()
    if not coordinate_passed and not near_ties:
        raise ValueError('Unexplained coordinate tolerance failure')
    delta = np.abs(scores-old_scores)
    exceeded = delta > (1e-5 + 1e-6*np.abs(old_scores))
    person, joint = np.unravel_index(delta.argmax(), delta.shape)
    return dict(strict_coordinate_tolerance_passed=coordinate_passed,
        near_tie_components=len(near_ties), near_ties=near_ties,
        historical_logit_parity_proven=False,
        coordinate_max_abs_error=float(np.abs(coords-old_coords).max()),
        score_max_abs_error=float(delta.max()), strict_score_atol=1e-5,
        strict_score_rtol=1e-6, strict_score_tolerance_passed=not bool(exceeded.any()),
        score_exceeded_elements=int(exceeded.sum()),
        score_threshold_changes=int(((scores > .2) != (old_scores > .2)).sum()),
        worst_score_difference=dict(annotation_id=int(annotation_ids[person]),
            joint=int(joint), cached=float(old_scores[person,joint]),
            current=float(scores[person,joint])))


def publish_npz(np, path, payload):
    """Publish only after exact, pickle-free roundtrip verification."""
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    with np.load(tmp, allow_pickle=False) as saved:
        for key, value in payload.items():
            np.testing.assert_array_equal(saved[key], value)
    os.replace(tmp, path)


def validate_committed(np, report_path, result_path, fingerprint, info, arrays):
    """Accept a resume shard only when provenance, bytes and identity match."""
    rep = json.loads(report_path.read_text())
    if (rep['fingerprint'] != fingerprint or rep['source'] != info
            or sha(result_path) != rep['result_sha256']):
        raise ValueError('Committed shard identity/hash mismatch: ' + str(result_path))
    with np.load(result_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['annotation_id'], arrays['annotation_id'])
        np.testing.assert_array_equal(saved['image_id'], arrays['image_id'])
        if (saved['gain'].shape != (info['count'], 4)
                or not np.isfinite(saved['gain']).all()):
            raise ValueError('Wrong/non-finite saved gain')
    return rep



def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache', 'ann', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--full', action='store_true')
    mode.add_argument('--sample-shards', type=int)
    mode.add_argument('--shard-index', type=int, help='Zero-based sorted source shard for diagnosis')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.sample_shards is not None and not 3 <= args.sample_shards <= 64:
        p.error('Sample mode accepts 3 to 64 shards; use --full for all people')
    for name in ('repo', 'cache', 'ann', 'checkpoint', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.output == args.cache or args.cache in args.output.parents:
        raise ValueError('Write labels outside the source cache directory')
    if sha(args.checkpoint) != WEIGHT_SHA:
        raise ValueError('Checkpoint SHA mismatch')
    train_sha = sha(args.ann)
    mmroot = args.repo / 'mmpose'
    sys.path[:0] = [str(mmroot), str(mmroot/'analysis/fyp')]
    os.chdir(mmroot)

    import numpy as np
    import torch
    from mmengine.config import Config
    from mmpose.apis import init_model
    from shared_cache import ReferenceCache, atomic_json
    from build_shared_cache import candidate_from_cache
    from oks_utility import compute_instance_oks, compute_gain

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run')
    cache = ReferenceCache(args.cache)
    cp = cache.run['protocol']
    if any(cp.get(k) != v for k, v in dict(dataset_split='train2017',
            bbox_source='gt', checkpoint_sha256=WEIGHT_SHA,
            annotation_sha256=train_sha, flip_test=False).items()):
        raise ValueError('Cache protocol mismatch')
    shards = cache.shards()
    if args.sample_shards is not None and len(shards) < args.sample_shards:
        raise ValueError('Not enough source shards')
    if args.shard_index is not None:
        if not 0 <= args.shard_index < len(shards):
            raise ValueError('Shard index out of range')
        indices = [args.shard_index]
    else:
        indices = (list(range(len(shards))) if args.full else
                   np.linspace(0, len(shards)-1, args.sample_shards, dtype=int).tolist())
    sources = [shards[i] for i in indices]
    source_info, ids = [], set()
    print('Validating source shards and hashes...', flush=True)
    for checked, (path, index) in enumerate(zip(sources, indices), 1):
        a = cache.load_shard(path)
        these = set(map(int, a['annotation_id']))
        if ids & these:
            raise ValueError('Repeated annotation across selected shards')
        ids.update(these)
        source_info.append(dict(name=path.name, source_shard_index=index,
            sha256=sha(path), count=len(a['annotation_id']),
            annotation_ids=a['annotation_id'].tolist()))
        if checked % 128 == 0 or checked == len(sources):
            print(f'Source check: {checked}/{len(sources)} shards, {len(ids)} people', flush=True)
    if args.full and (ids != set(cache.expected) or len(ids) != 149813
                      or cp.get('scope') != 'full'):
        raise ValueError('Full train coverage must exactly match all 149813 approved people')
    del a
    cfg_path = mmroot / ('projects/cigpose/body_2d_keypoint/'
                         'cigpose-m_1xb384-420e_coco-256x192.py')
    cfg = Config.fromfile(str(cfg_path))
    cfg.model.test_cfg['flip_test'] = False
    cfg_hash = digest(cfg.to_dict())
    codefiles = [Path(__file__).resolve(), Path(__file__).resolve().with_name('config.py'), Path(__file__).resolve().with_name('common.py'),
        mmroot/'analysis/fyp/build_shared_cache.py', mmroot/'analysis/fyp/shared_cache.py',
        mmroot/'analysis/fyp/oks_utility.py',
        mmroot/'mmpose/models/heads/coord_cls_heads/cig_head.py',
        mmroot/'mmpose/models/pose_estimators/topdown.py']
    commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    run = dict(schema_version='b-only-labels-v4', development_sample_only=not args.full,
        scope='full' if args.full else 'sample',
        git_commit=commit, source_cache=str(args.cache),
        source_cache_run_sha256=sha(args.cache/'run.json'), source_cache_protocol=cp,
        checkpoint_sha256=WEIGHT_SHA, annotation_sha256=train_sha,
        source_shards=source_info, selected_instances=len(ids),
        selection='uncertainty_top1', alphas=list(STRENGTHS), always_accept=True,
        sampling=('all approved train GT people' if args.full else
                  'specified source shard' if args.shard_index is not None else
                  'evenly spaced whole shards, including first and last; not random'),
        supervision_reference='P0 recomputed from unchanged cached features under current runtime',
        old_cache_score_policy='Audit against original 1e-5/1e-6 tolerance; not a label gate. No AP parity claim.',
        old_cache_coordinate_policy='0.1px strict audit; only raw-bin near ties may continue; all unexplained differences stop',
        near_tie_policy=dict(bin_distance_policy='record without adjacency assumption', bin_roundtrip_atol=1e-3,
            logit_gap_bound='8*float32_eps*max(1,abs(current),abs(old))',
            discrete_decoder_only=True, exact_current_bypass_required=True,
            historical_logits_unavailable=True, preserves_raw_predictions=True),
        config_sha256=cfg_hash, code_sha256={str(f): sha(f) for f in codefiles},
        versions={k: importlib.metadata.version(k) for k in
                  ['torch', 'torchvision', 'numpy', 'mmcv', 'mmengine', 'mmpose']},
        python=sys.version, runtime=dict(seed=42, deterministic_algorithms=True,
            CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False,
            cudnn_deterministic=True, precision='float32', flip_test=False,
            base_eval=True, base_frozen=True, device=torch.cuda.get_device_name(0)),
        tie_policy='Exact gain ties choose smallest alpha; retain all gains.',
        label_layout={'gain': '[B,4], float64; columns follow alphas',
            'candidate_coords': '[B,4,17,2], original image coordinates',
            'candidate_scores': '[B,4,17]',
            'annotation_id': '[B], join to source train cache by ID',
            'best_alpha_index': '[B], offline oracle diagnostic, not inference input'},
        limitations=['Label generation only: no training or formal AP/AR.',
            'GT and true gains are offline supervision, never inference inputs.',
            'Old cache metadata preserved, current settings recorded separately.',
            'Historical cached-score tolerance/threshold changes are reported, not silently passed.',
            'Near ties are a diagnostic explanation, not proof of historical logit parity.',
            'Formal AP/AR scoring and postprocessing are unchanged; their parity needs separate validation.'])
    fingerprint = digest(run)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Output exists; use --resume for the identical run')
        if json.loads((args.output/'run.json').read_text()) != run:
            raise ValueError('Resume code/protocol/source/versions differ; use a new output')
    else:
        if args.resume:
            raise FileNotFoundError('Resume requested but output does not exist')
        args.output.mkdir(parents=True)
        atomic_json(args.output/'run.json', run)
    lock = args.output/'.writer.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        cfg.dump(str(args.output/'resolved_config.py'))
        model = None
        anns = None
        reports = []
        for seq, (source_path, info) in enumerate(zip(sources, source_info), 1):
            if sha(source_path) != info['sha256']:
                raise ValueError('Source file changed during the run: ' + str(source_path))
            a = cache.load_shard(source_path)
            stem = Path(info['name']).stem
            result_path = args.output/(stem+'.npz')
            report_path = args.output/(stem+'.json')
            if report_path.exists():
                rep = validate_committed(np, report_path, result_path, fingerprint, info, a)
                reports.append(rep)
                print(f'[{seq}/{len(sources)}] verified; skip {stem}', flush=True)
                continue
            if model is None:
                model = init_model(cfg, str(args.checkpoint), device='cuda:0')
                model.requires_grad_(False).eval()
                data = json.loads(args.ann.read_text(encoding='utf-8-sig'))
                anns = {int(v['id']): v for v in data['annotations'] if int(v['id']) in ids}
                del data
            n = len(a['annotation_id'])
            originals = {k: v.copy() for k, v in a.items()}
            gt, areas = [], []
            for image_id, ann_id in zip(a['image_id'], a['annotation_id']):
                ann = anns[int(ann_id)]
                if (ann['image_id'] != int(image_id) or ann['category_id'] != 1
                        or ann.get('iscrowd', 0) != 0):
                    raise ValueError('GT identity mismatch')
                gt.append(np.asarray(ann['keypoints'], dtype=np.float64).reshape(17, 3))
                areas.append(float(ann['area']))
            gt, areas = np.asarray(gt), np.asarray(areas)
            cached_oks = np.asarray([compute_instance_oks(a['P_0_coords'][i], gt[i], areas[i])
                                     for i in range(n)])
            np.testing.assert_array_equal(a['top1_joint'], a['uncertainty'].argmax(axis=1))
            joints = torch.as_tensor(a['top1_joint'], dtype=torch.long, device='cuda:0')

            def candidate(alpha):
                with torch.inference_mode():
                    c, s = candidate_from_cache(model.head, a, joints,
                        torch.full((n,), alpha, device='cuda:0'), 'cuda:0')
                if c.shape != (n, 17, 2) or s.shape != (n, 17):
                    raise ValueError('Candidate shape mismatch')
                if not np.isfinite(c).all() or not np.isfinite(s).all():
                    raise ValueError('Non-finite output')
                for k in a:
                    np.testing.assert_array_equal(a[k], originals[k])
                return c, s

            def compare(c, s, rc, rs):
                np.testing.assert_array_equal(c, rc)
                np.testing.assert_array_equal(s, rs)
                np.testing.assert_array_equal(s > .2, rs > .2)
                return dict(coordinate_max_abs_error=float(np.abs(c-rc).max()),
                            score_max_abs_error=float(np.abs(s-rs).max()))

            c0, s0 = candidate(0.)
            legacy = audit_legacy_reference(np, c0, s0, a['P_0_coords'],
                                            a['P_0_scores'], a['annotation_id'],
                lambda: verify_near_ties(np, torch, model, a, c0, s0))
            before = np.asarray([compute_instance_oks(c0[i], gt[i], areas[i])
                                 for i in range(n)])
            legacy['max_abs_oks_difference'] = float(np.abs(before-cached_oks).max())
            again_c0, again_s0 = candidate(0.)
            alpha0 = compare(again_c0, again_s0, c0, s0)
            alpha0['max_abs_gain'] = max(abs(compute_gain(again_c0[i], c0[i],
                                            gt[i], areas[i])) for i in range(n))
            # Independent existing full-replacement API, grouped by selected joint.
            with torch.inference_mode():
                features = torch.as_tensor(a['features'], device='cuda:0')
                base_copy = features.clone()
                changed, _ = model.head.cim.apply_intervention(features, joints,
                                                    torch.ones(n, device='cuda:0'))
                for joint in torch.unique(joints).tolist():
                    mask = joints == joint
                    forced, _ = model.head.cim.force_intervene(features[mask], joint)
                    if not torch.equal(changed[mask], forced):
                        raise ValueError('alpha=1 differs from independent full replacement')
                if not torch.equal(features, base_copy):
                    raise ValueError('Source GPU features modified')
            candidates = {alpha: candidate(alpha) for alpha in STRENGTHS}
            order_checks = {str(alpha): compare(*candidate(alpha), *candidates[alpha])
                            for alpha in reversed(STRENGTHS)}
            after = np.empty((n, 4), dtype=np.float64)
            gains = np.empty_like(after)
            for j, alpha in enumerate(STRENGTHS):
                for i in range(n):
                    after[i, j] = compute_instance_oks(candidates[alpha][0][i], gt[i], areas[i])
                    gains[i, j] = compute_gain(candidates[alpha][0][i], c0[i],
                                               gt[i], areas[i])
            np.testing.assert_allclose(gains, after-before[:, None], atol=1e-12, rtol=0)
            best = gains.argmax(axis=1)
            payload = dict(annotation_id=a['annotation_id'], image_id=a['image_id'],
                joint_id=a['top1_joint'], alphas=np.asarray(STRENGTHS, dtype=np.float32),
                gt_keypoints=gt, gt_area=areas, P_0_coords=c0,
                P_0_scores=s0, cached_P_0_coords=a['P_0_coords'],
                cached_P_0_scores=a['P_0_scores'], cached_oks_before=cached_oks, oks_before=before, oks_after=after,
                gain=gains, best_alpha_index=best,
                candidate_coords=np.stack([candidates[x][0] for x in STRENGTHS], axis=1),
                candidate_scores=np.stack([candidates[x][1] for x in STRENGTHS], axis=1))
            publish_npz(np, result_path, payload)
            rep = dict(fingerprint=fingerprint, source=info, instances=n, candidate_records=n*4,
                alpha0=alpha0, legacy_reference_audit=legacy,
                alpha1_full_replacement_features_equal=True,
                candidate_order_checks=order_checks, source_arrays_unchanged=True,
                result_sha256=sha(result_path),
                gain_stats={str(alpha): dict(sum=float(gains[:, j].sum()),
                    min=float(gains[:, j].min()), max=float(gains[:, j].max()),
                    positive=int((gains[:, j] > 0).sum()), negative=int((gains[:, j] < 0).sum()),
                    zero=int((gains[:, j] == 0).sum())) for j, alpha in enumerate(STRENGTHS)},
                oracle_counts={str(alpha): int((best == j).sum()) for j, alpha in enumerate(STRENGTHS)},
                oracle_gain_sum=float(gains.max(axis=1).sum()),
                all_negative=int((gains.max(axis=1) < 0).sum()),
                exact_best_ties=int(((gains == gains.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum()))
            atomic_json(report_path, rep)  # Commit marker: publish after NPZ verified.
            reports.append(rep)
            tag = '' if legacy['strict_score_tolerance_passed'] else ' [old-cache score tolerance exceeded: audited]'
            if not legacy['strict_coordinate_tolerance_passed']:
                tag += ' [old-cache coordinate tolerance FAILED: near tie audited]'
            print(f'[{seq}/{len(sources)}] LABEL CHECKS PASS {stem}: {n} people, {n*4} labels' + tag, flush=True)
        count = sum(r['instances'] for r in reports)
        if count != len(ids):
            raise ValueError('Final sampled ID coverage mismatch')
        summary = dict(passed=True, development_sample_only=not args.full, fingerprint=fingerprint,
            scope='full' if args.full else 'sample',
            supervision_reference=run['supervision_reference'],
            old_cache_strict_coordinate_tolerance_passed=all(
                r['legacy_reference_audit']['strict_coordinate_tolerance_passed'] for r in reports),
            old_cache_near_tie_components=sum(
                r['legacy_reference_audit']['near_tie_components'] for r in reports),
            old_cache_near_tie_shards=sum(
                not r['legacy_reference_audit']['strict_coordinate_tolerance_passed'] for r in reports),
            old_cache_strict_score_tolerance_passed=all(r['legacy_reference_audit']['strict_score_tolerance_passed'] for r in reports),
            old_cache_score_exceeded_elements=sum(r['legacy_reference_audit']['score_exceeded_elements'] for r in reports),
            old_cache_score_threshold_changes=sum(r['legacy_reference_audit']['score_threshold_changes'] for r in reports),
            old_cache_score_max_error=max(r['legacy_reference_audit']['score_max_abs_error'] for r in reports),
            old_cache_coordinate_max_error=max(r['legacy_reference_audit']['coordinate_max_abs_error'] for r in reports),
            old_cache_max_abs_oks_difference=max(r['legacy_reference_audit']['max_abs_oks_difference'] for r in reports),
            oracle_mean_gain=sum(r['oracle_gain_sum'] for r in reports)/count,
            all_negative_people=sum(r['all_negative'] for r in reports),
            exact_best_tie_people=sum(r['exact_best_ties'] for r in reports),
            selected_shards=len(sources), instances=count, candidate_records=count*4,
            alpha0_max_abs_gain=max(r['alpha0']['max_abs_gain'] for r in reports),
            alpha0_max_coordinate_error=max(r['alpha0']['coordinate_max_abs_error'] for r in reports),
            alpha0_max_score_error=max(r['alpha0']['score_max_abs_error'] for r in reports),
            alpha1_full_replacement_features_equal=True, source_arrays_unchanged=True,
            gain_summary={str(alpha): dict(
                mean_gain=sum(r['gain_stats'][str(alpha)]['sum'] for r in reports)/count,
                positive=sum(r['gain_stats'][str(alpha)]['positive'] for r in reports),
                negative=sum(r['gain_stats'][str(alpha)]['negative'] for r in reports),
                zero=sum(r['gain_stats'][str(alpha)]['zero'] for r in reports)) for alpha in STRENGTHS},
            oracle_best_alpha_counts={str(alpha): sum(r['oracle_counts'][str(alpha)] for r in reports)
                                     for alpha in STRENGTHS})
        summary['best_fixed_alpha'] = max(summary['gain_summary'],
            key=lambda k: summary['gain_summary'][k]['mean_gain'])
        summary['oracle_improvement_over_best_fixed'] = summary['oracle_mean_gain'] - max(
            v['mean_gain'] for v in summary['gain_summary'].values())
        atomic_json(args.output/'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
