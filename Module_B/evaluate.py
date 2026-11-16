"""Evaluate B and controls on the fixed val2017 detection cache."""
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import contextlib
import importlib.metadata
import json
import random
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from Module_B.config import (ALPHAS, FEATURE_SPEC, WEIGHT_SHA, EXPECTED_DETECTIONS,
                             FIXED_BOX_SOURCE, ORIGINAL_TRAIN_SHA, NETWORK)
from Module_B.common import sha, digest, atomic_json, source_hashes
from Module_B.data import make_features
from Module_B.model import get_model

MODES = ('p0_current', 'fixed_alpha025', 'top1_alpha1_current', 'b_current')

def find_detection_cache(roots):
    """Bounded discovery tolerating inaccessible Windows/WSL filesystem entries.

    Keep metadata opens and directory iteration inside OSError handlers: pathlib
    glob itself can fail on invalid WpSystem entries before yielding a path.
    """
    matches = set()
    skipped = set()
    system_dirs = {'wpsystem', 'system volume information', '$recycle.bin',
                   'recovery', '.git'}
    for root in roots:
        try:
            root = Path(root).resolve()
        except OSError:
            skipped.add(str(root))
            continue
        stack = [(root, 0)]
        while stack:
            directory, depth = stack.pop()
            try:
                run = json.loads((directory/'run.json').read_text(encoding='utf-8'))
            except FileNotFoundError:
                run = None
            except (OSError, UnicodeError, ValueError):
                skipped.add(str(directory/'run.json'))
                run = None
            if isinstance(run, dict):
                cp = run.get('protocol', {})
                if (isinstance(cp, dict)
                        and run.get('schema_version') == 'cigpose-detection-cache-v1'
                        and cp.get('dataset_split') == 'val2017'
                        and (cp.get('scope') == 'full' or
                             (cp.get('scope') is None and
                              len(run.get('expected_pairs', [])) == EXPECTED_DETECTIONS))):
                    matches.add(directory)
            if depth >= 3:
                continue
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.name.lower() in system_dirs:
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append((Path(entry.path), depth+1))
                        except OSError:
                            skipped.add(entry.path)
            except OSError:
                skipped.add(str(directory))
    if skipped:
        print('Auto-cache search skipped inaccessible/invalid entries:',len(skipped),flush=True)
    if len(matches) != 1:
        raise ValueError('Need exactly one full val2017 detection cache; found '
                         +str(len(matches))+': '+str(sorted(map(str,matches)))
                         +'. Supply the intended directory with --cache PATH.')
    return matches.pop()


def validate_detection_protocol(cache, checkpoint_sha):
    """Older detection-cache metadata has no scope field.

    An absent scope is not a completeness claim. Require the agreed fixed-box
    protocol and 104125 unique expected identities, then verify(True) must check
    every committed source shard against that exact manifest before inference.
    """
    cp = cache.run['protocol']
    if (cp.get('dataset_split') != 'val2017' or cp.get('scope') not in (None, 'full')
            or cp.get('bbox_source') != FIXED_BOX_SOURCE
            or len(cache.expected) != EXPECTED_DETECTIONS
            or cp.get('flip_test') is not False
            or not cp.get('source_detection_sha256')
            or cp.get('checkpoint_sha256') != WEIGHT_SHA or checkpoint_sha != WEIGHT_SHA):
        raise ValueError('Detection cache protocol mismatch: '+str(dict(
            dataset_split=cp.get('dataset_split'), scope=cp.get('scope'),
            bbox_source=cp.get('bbox_source'), expected_instances=len(cache.expected),
            flip_test=cp.get('flip_test'), checkpoint_sha256=cp.get('checkpoint_sha256'),
            source_detection_sha256_present=bool(cp.get('source_detection_sha256')))))


def validate_b_run(folder, trainer_path):
    run_path, summary_path = folder/'run.json', folder/'summary.json'
    run = json.loads(run_path.read_text())
    summary = json.loads(summary_path.read_text())
    checkpoint = folder/'best.pt'
    if (run.get('schema_version') != 'b-gain-regressor-v1' or run.get('scope') != 'full'
            or run.get('feature_spec') != FEATURE_SPEC or run.get('alphas') != ALPHAS
            or run.get('script_sha256') not in (sha(trainer_path), ORIGINAL_TRAIN_SHA)
            or run.get('network') != NETWORK
            or summary.get('run_fingerprint') != digest(run)
            or summary.get('training_completed') is not True
            or summary.get('scope') != 'full' or summary.get('val2017_used') is not False
            or summary.get('gt_in_inputs') is not False
            or summary.get('best_checkpoint_sha256') != sha(checkpoint)):
        raise ValueError('B best checkpoint/training provenance/input schema mismatch')
    recorded = run.get('code_sha256')
    if recorded is not None and recorded != source_hashes('train.py','common.py','config.py','data.py','model.py'):
        raise ValueError('Training dependency code changed')
    return run, summary


def records(cache, arrays, coords, scores):
    for i, idx in enumerate(arrays['source_detection_index']):
        yield dict(image_id=int(arrays['image_id'][i]),
            detection_instance_id=cache.instance_id(idx), source_detection_index=int(idx),
            category_id=1, bbox=arrays['bbox'][i].tolist(), bbox_format='xyxy',
            bbox_score=float(arrays['bbox_score'][i]), input_scale=arrays['input_scale'][i].tolist(),
            keypoints=coords[i].tolist(), keypoint_scores=scores[i].tolist())


def verify_prediction_payload(arrays, saved):
    for k in ('source_detection_index','image_id','bbox','bbox_score','input_scale'):
        np.testing.assert_array_equal(saved[k], arrays[k])
    n = len(arrays['source_detection_index'])
    if (saved['predicted_gain'].shape != (n,4) or saved['selected_alpha'].shape != (n,)
            or not np.isin(saved['selected_alpha'], ALPHAS).all()):
        raise ValueError('B prediction/selection shape mismatch')
    np.testing.assert_array_equal(saved['selected_alpha'],
        np.asarray(ALPHAS,np.float32)[saved['predicted_gain'].argmax(axis=1)])
    for mode in MODES:
        if saved[mode+'_coords'].shape != (n,17,2) or saved[mode+'_scores'].shape != (n,17):
            raise ValueError('Wrong predicted pose shape: '+mode)
    if not all(np.isfinite(saved[k]).all() for k in saved):
        raise ValueError('Non-finite prediction output')


def save_npz(path,payload):
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**payload);f.flush();os.fsync(f.fileno())
    with np.load(tmp,allow_pickle=False) as data:
        for k in payload:
            np.testing.assert_array_equal(data[k],payload[k])
    os.replace(tmp,path)


def evaluate_mode(cache,sources,output,ann,mode,meta):
    from mmpose.evaluation.metrics import CocoMetric
    from eval_unified import UnifiedPoseEvaluator
    folder=output/mode;folder.mkdir(exist_ok=True)
    metric=CocoMetric(ann_file=str(ann),outfile_prefix=str(folder/'predictions'),
        score_mode='bbox_keypoint',keypoint_score_thr=.2,nms_mode='oks_nms',nms_thr=.9)
    metric.dataset_meta=meta
    cpu=UnifiedPoseEvaluator(ann)
    n=0
    with (folder/'raw_predictions_before_nms.json').open('w') as out:
        out.write('[')
        for source in sources:
            a=cache.load_shard(source)
            with np.load(output/'predictions'/source.name,allow_pickle=False) as saved:
                data={k:saved[k] for k in saved.files}
            verify_prediction_payload(a,data)
            rows=list(records(cache,a,data[mode+'_coords'],data[mode+'_scores']))
            for r in rows:
                if n:out.write(',\n')
                json.dump(r,out,allow_nan=False);n+=1
                metric.process([], [dict(id=r['source_detection_index'],img_id=r['image_id'],
                    pred_instances=dict(keypoints=np.asarray([r['keypoints']],np.float32),
                        keypoint_scores=np.asarray([r['keypoint_scores']],np.float32),
                        bboxes=np.asarray([r['bbox']],np.float32),
                        bbox_scores=np.asarray([r['bbox_score']],np.float32)),
                    gt_instances=dict(bbox_scales=np.asarray([r['input_scale']],np.float32)))])
            cpu.add_records(rows)
        out.write(']\n')
    with (folder/'evaluation.log').open('w') as log,contextlib.redirect_stdout(log):
        official=dict(metric.evaluate(n))
        pre,audit=cpu.evaluate()
        final_rows=json.loads((folder/'predictions.keypoints.json').read_text())
        final,final_audit=cpu.evaluate_coco_results(final_rows)
    if set(official)!=set(pre) or set(pre)!=set(final) or len(official)!=10:
        raise ValueError('Different official/public metric names')
    for k in official:
        np.testing.assert_allclose(official[k],pre[k],rtol=0,atol=1e-12)
        np.testing.assert_allclose(official[k],final[k],rtol=0,atol=1e-12)
    if len(cpu.last_coco_results)!=len(final_rows):
        raise ValueError('Postprocessing output count mismatch')
    for a,b in zip(cpu.last_coco_results,final_rows):
        if a['image_id']!=b['image_id']:
            raise ValueError('Postprocessing image order mismatch')
        np.testing.assert_array_equal(a['keypoints'],b['keypoints'])
        np.testing.assert_array_equal(a['bbox'],b['bbox'])
        np.testing.assert_allclose(a['score'],b['score'],rtol=0,atol=1e-12)
    if audit['evaluation_image_count']!=5000 or final_audit['evaluation_image_count']!=5000:
        raise ValueError('COCO evaluation must cover all 5000 val2017 images')
    if n!=len(cache.expected):
        raise ValueError('Incomplete detection coverage at scoring')
    result=dict(reference_metrics=official,cpu_pre_nms_metrics=pre,cpu_final_metrics=final,
        audit=audit,input_count=n,all_10_metrics_match=True,postprocessing_records_match=True,
        final_predictions_sha256=sha(folder/'predictions.keypoints.json'))
    atomic_json(folder/'metrics.json',result)
    print(mode,official,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','ann','checkpoint','b-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cache',required=True,help='Detection cache directory, or auto')
    p.add_argument('--search-root',type=Path,action='append',default=[])
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--full',action='store_true')
    mode.add_argument('--smoke-shards',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.smoke_shards is not None and args.smoke_shards<1:
        p.error('--smoke-shards must be positive')
    for name in ('repo','ann','checkpoint','b_run','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.cache=='auto':
        if not args.search_root:p.error('auto cache requires --search-root')
        args.cache=find_detection_cache(args.search_root)
    else:args.cache=Path(args.cache).resolve()
    print('Detection cache:',args.cache,flush=True)
    for protected in (args.cache,args.b_run):
        if args.output==protected or protected in args.output.parents or args.output in protected.parents:
            raise ValueError('Output must be separate from cache and B training artifacts')
    mmroot=args.repo/'mmpose'
    sys.path[:0]=[str(args.repo/'scripts'),str(mmroot),str(mmroot/'analysis/fyp')]
    os.chdir(mmroot)
    from detection_cache import ReferenceCache
    from build_shared_cache import candidate_from_cache
    from mmengine.config import Config
    from mmpose.apis import init_model
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cache=ReferenceCache(args.cache)
    cp=cache.run['protocol']
    validate_detection_protocol(cache,sha(args.checkpoint))
    ann_sha=sha(args.ann)
    if cp.get('annotation_sha256',ann_sha)!=ann_sha:
        raise ValueError('Validation annotation differs from source-cache provenance')
    annotation_meta=json.loads(args.ann.read_text(encoding='utf-8-sig'))
    val_images={int(v['id']) for v in annotation_meta['images']}
    if len(val_images)!=5000 or not set(cache.expected.values())<=val_images:
        raise ValueError('Expected full COCO val2017 annotation image set')
    del annotation_meta
    b_run,b_summary=validate_b_run(args.b_run,Path(__file__).with_name('train.py'))
    split_path=args.b_run/'split.json'
    split=json.loads(split_path.read_text())
    if digest(split)!=b_run['split_sha256']:
        raise ValueError('B training split manifest changed')
    train_images=set(split['train_image_ids'])|set(split['development_image_ids'])
    if train_images & val_images:
        raise ValueError('Train/development and val2017 images overlap')
    sources=cache.shards()
    cache.verify(True)
    print(f'Complete detection cache verified: {len(cache.expected)} instances',flush=True)
    if args.smoke_shards is not None:
        if args.smoke_shards>len(sources):raise ValueError('Too few source shards')
        indices=np.linspace(0,len(sources)-1,args.smoke_shards,dtype=int)
        sources=[sources[int(i)] for i in indices]
    source_info=[dict(name=f.name,sha256=sha(f)) for f in sources]
    cfg_path=mmroot/'projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py'
    cfg=Config.fromfile(str(cfg_path));cfg.model.test_cfg['flip_test']=False
    if cp.get('config_sha256',digest(cfg.to_dict()))!=digest(cfg.to_dict()):
        raise ValueError('Model config differs from detection cache')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    code=[Path(__file__).resolve(),Path(__file__).with_name('train.py'), Path(__file__).with_name('common.py'), Path(__file__).with_name('config.py'), Path(__file__).with_name('data.py'), Path(__file__).with_name('model.py'),
        mmroot/'analysis/fyp/detection_cache.py',mmroot/'analysis/fyp/shared_cache.py',
        mmroot/'analysis/fyp/build_shared_cache.py',mmroot/'analysis/fyp/eval_unified.py',
        mmroot/'mmpose/models/heads/coord_cls_heads/cig_head.py',cfg_path]
    run=dict(schema_version='b-detection-evaluation-v1',scope='full' if args.full else 'smoke',
        source_cache=str(args.cache),source_cache_run_sha256=sha(args.cache/'run.json'),
        source_protocol=cp,source_shards=source_info,checkpoint_sha256=WEIGHT_SHA,
        b_checkpoint_sha256=sha(args.b_run/'best.pt'),b_training_fingerprint=digest(b_run),
        b_training_summary_sha256=sha(args.b_run/'summary.json'),annotation_sha256=ann_sha,
        code_sha256={str(f):sha(f) for f in code},config_sha256=digest(cfg.to_dict()),
        modes=list(MODES),alphas=ALPHAS,seed=42,deterministic=True,matmul_tf32=False,
        cudnn_tf32=True,flip_test=False,device=torch.cuda.get_device_name(0),
        versions={k:importlib.metadata.version(k) for k in ['torch','numpy','mmpose','mmcv','mmengine','pycocotools']},
        scoring=dict(score_mode='bbox_keypoint',keypoint_score_thr=.2,nms_mode='oks_nms',nms_thr=.9),
        historical_cache_parity_claim=False,
        reference='All four evaluated methods recomputed from the same unchanged detection features',
        cache_completeness=dict(expected_instances=EXPECTED_DETECTIONS,
            all_source_shards_verified=True,metadata_scope=cp.get('scope'),
            basis='exact fixed-box identity manifest and complete shard coverage'),
        gt_used_for_action=False,gt_role='COCO scoring only; image identities also checked for split isolation')
    fingerprint=digest(run)
    if args.output.exists():
        if not args.resume:raise FileExistsError('Output exists; use identical --resume or new directory')
        if json.loads((args.output/'run.json').read_text())!=run:raise ValueError('Resume provenance changed')
    else:
        if args.resume:raise FileNotFoundError('Resume output absent')
        args.output.mkdir(parents=True);atomic_json(args.output/'run.json',run)
    pred_dir=args.output/'predictions';pred_dir.mkdir(exist_ok=True)
    lock=args.output/'.writer.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        model=init_model(cfg,str(args.checkpoint),device='cuda:0');model.requires_grad_(False).eval()
        b=get_model('cuda:0')
        state=torch.load(args.b_run/'best.pt',map_location='cpu',weights_only=True)
        if (state['run_fingerprint']!=digest(b_run) or state['feature_spec']!=FEATURE_SPEC
                or state['alphas']!=ALPHAS or state['epoch']!=b_summary['best_epoch']
                or state['gain_scale']!=b_run['gain_scale']):raise ValueError('Checkpoint schema/epoch mismatch')
        mean=state['normalization_mean'].numpy();std=state['normalization_std'].numpy()
        if mean.shape!=(563,) or std.shape!=(563,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
            raise ValueError('Invalid normalization statistics')
        b.load_state_dict(state['model']);b.requires_grad_(False).eval()
        seen=set();reports=[]
        for seq,(source,info) in enumerate(zip(sources,source_info),1):
            if sha(source)!=info['sha256']:raise ValueError('Source cache changed')
            a=cache.load_shard(source)
            source_ids=set(map(int,a['source_detection_index']))
            if seen&source_ids:raise ValueError('Duplicate detection IDs')
            seen.update(source_ids)
            out=pred_dir/source.name;rp=out.with_suffix('.json')
            if rp.exists():
                rep=json.loads(rp.read_text())
                if rep['fingerprint']!=fingerprint or rep['source']!=info or sha(out)!=rep['result_sha256']:
                    raise ValueError('Saved prediction identity/hash mismatch')
                with np.load(out,allow_pickle=False) as saved:
                    verify_prediction_payload(a,{k:saved[k] for k in saved.files})
                reports.append(rep);print(f'[{seq}/{len(sources)}] verified; skip {source.stem}',flush=True)
                continue
            original={k:v.copy() for k,v in a.items()}
            n=len(a['source_detection_index']);joints=torch.as_tensor(a['top1_joint'],device='cuda:0')
            def candidate(alpha):
                action=np.broadcast_to(np.asarray(alpha,dtype=np.float32),(n,)).copy()
                with torch.inference_mode():
                    return candidate_from_cache(model.head,a,joints,torch.as_tensor(action,device='cuda:0'),'cuda:0')
            features=(make_features(a)-mean)/std
            if not np.isfinite(features).all():raise ValueError('Non-finite B input')
            with torch.inference_mode():
                predicted=b(torch.from_numpy(features).cuda()).cpu().numpy()/state['gain_scale']
                repeated=b(torch.from_numpy(features).cuda()).cpu().numpy()/state['gain_scale']
            np.testing.assert_array_equal(predicted,repeated)
            chosen=np.asarray(ALPHAS,np.float32)[predicted.argmax(axis=1)]
            poses=dict(zip(MODES,[candidate(0.),candidate(.25),candidate(1.),candidate(chosen)]))
            np.testing.assert_array_equal(candidate(0.)[0],poses['p0_current'][0])
            np.testing.assert_array_equal(candidate(0.)[1],poses['p0_current'][1])
            c,s=candidate(chosen)
            np.testing.assert_array_equal(c,poses['b_current'][0]);np.testing.assert_array_equal(s,poses['b_current'][1])
            # Smoke verifies mixed-action B output against every uniform-alpha candidate.
            if not args.full:
                for alpha in ALPHAS:
                    mask=chosen==alpha
                    if mask.any():
                        cc,ss=candidate(alpha)
                        np.testing.assert_array_equal(c[mask],cc[mask]);np.testing.assert_array_equal(s[mask],ss[mask])
            for k in a:np.testing.assert_array_equal(a[k],original[k])
            payload={k:a[k] for k in ('source_detection_index','image_id','bbox','bbox_score','input_scale')}
            payload.update(predicted_gain=predicted,selected_alpha=chosen)
            for mode,(coords,scores) in poses.items():
                payload[mode+'_coords']=coords;payload[mode+'_scores']=scores
            verify_prediction_payload(a,payload);save_npz(out,payload)
            rep=dict(fingerprint=fingerprint,source=info,result_sha256=sha(out),instances=n,
                source_arrays_unchanged=True,current_repeats_exact=True,
                selected_alpha_counts={str(v):int((chosen==v).sum()) for v in ALPHAS},
                historical_p0_coordinate_max_error=float(np.abs(poses['p0_current'][0]-a['P_0_coords']).max()),
                historical_p0_score_max_error=float(np.abs(poses['p0_current'][1]-a['P_0_scores']).max()),
                historical_top1_coordinate_max_error=float(np.abs(poses['top1_alpha1_current'][0]-a['top1_coords']).max()))
            atomic_json(rp,rep);reports.append(rep)
            print(f'[{seq}/{len(sources)}] PREDICTION CHECKS PASS {source.stem}: {n} detections',flush=True)
        summary=dict(prediction_checks_passed=True,scope=run['scope'],fingerprint=fingerprint,
            detection_cache=str(args.cache),detections=len(seen),shards=len(sources),
            b_best_epoch=b_summary['best_epoch'],current_repeats_exact=True,source_arrays_unchanged=True,
            selected_alpha_counts={str(v):sum(r['selected_alpha_counts'][str(v)] for r in reports) for v in ALPHAS},
            historical_p0_coordinate_max_error=max(r['historical_p0_coordinate_max_error'] for r in reports),
            historical_p0_score_max_error=max(r['historical_p0_score_max_error'] for r in reports),
            historical_cache_parity_claim=False,formal_coco_ap_evaluated=False)
        if args.full:
            if seen!=set(cache.expected):raise ValueError('Incomplete full detection coverage')
            from mmpose.datasets.datasets.utils import parse_pose_metainfo
            from mmengine.logging import MMLogger
            MMLogger.get_current_instance()
            meta=parse_pose_metainfo(dict(from_file='configs/_base_/datasets/coco.py'))
            scores={mode:evaluate_mode(cache,sources,args.output,args.ann,mode,meta) for mode in MODES}
            summary['evaluations']=scores
            summary['formal_coco_ap_evaluated']=True
            ap={mode:float(scores[mode]['reference_metrics']['coco/AP']) for mode in MODES}
            summary['ap_0_to_1']=ap
            summary['b_minus_controls_ap_points']={mode:100*(ap['b_current']-ap[mode]) for mode in MODES if mode!='b_current'}
        atomic_json(args.output/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    finally:lock.unlink()


if __name__=='__main__':main()
