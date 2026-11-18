"""Re-evaluate fixed P0 and freshly infer Top-1/original under public V2.

No training and no cache writes. Raw predictions stay outside Git. Original runs
from images; Top-1 runs from the same cached features used by A/B/C.
"""
import argparse
import contextlib
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from build_detection_cache import DetectionDataset, load_detection_instances
from build_shared_cache import EXPECTED_CHECKPOINT, array, candidate_from_cache
from detection_cache import ReferenceCache
from evaluate_detection_cache import records_from_arrays
from eval_unified import UnifiedPoseEvaluator
from precision_policy import configure_legacy_precision, runtime_record
from shared_cache import atomic_json, hash_file

ROOT = Path(__file__).resolve().parents[3]


def load_base(checkpoint, device):
    sys.path.insert(0, str(ROOT / 'mmpose'))
    from mmengine.config import Config
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    register_all_modules()
    previous = Path.cwd()
    try:
        os.chdir(ROOT / 'mmpose')
        cfg = Config.fromfile('projects/cigpose/body_2d_keypoint/cigpose-m_coco-256x192_detection-single-view.py')
        assert cfg.model.test_cfg.flip_test is False
        assert tuple(cfg.codec.input_size) == (192, 256)
        meta = parse_pose_metainfo(dict(from_file='configs/_base_/datasets/coco.py'))
        base = init_model(cfg, str(checkpoint), device=device).requires_grad_(False).eval()
    finally:
        os.chdir(previous)
    return base, cfg, meta


def original_batches(records, cfg, meta, workers):
    from torch.utils.data import DataLoader
    from mmengine.dataset import pseudo_collate
    dataset = DetectionDataset(records, cfg.val_pipeline, meta)
    return iter(DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers,
                           collate_fn=pseudo_collate))


@torch.inference_mode()
def run_mode(mode, args, cache, base, cfg, meta, records):
    from mmpose.evaluation.metrics import CocoMetric
    folder = args.output / mode
    folder.mkdir()  # Do not overwrite a previous run or partial result.
    started = time.monotonic()
    paths = cache.shards()
    if args.limit_batches:
        paths = paths[:args.limit_batches]
    loader = original_batches(records, cfg, meta, args.workers) if mode == 'original' else None
    with (folder / 'evaluation.log').open('w') as log, contextlib.redirect_stdout(log):
        cpu = UnifiedPoseEvaluator(args.ann)
        official = CocoMetric(ann_file=str(args.ann), outfile_prefix=str(folder / 'predictions'),
                              score_mode='bbox_keypoint', keypoint_score_thr=.2,
                              nms_mode='oks_nms', nms_thr=.9)
        official.dataset_meta = meta
    seen = set()
    count = 0
    delta = dict(changed_instances=0, max_coordinate_difference=0., max_score_difference=0.)
    with (folder / 'raw_predictions_before_nms.json').open('w') as raw:
        raw.write('[')
        for step, path in enumerate(paths, 1):
            stored = cache.load_shard(path)
            arrays = dict(stored)
            ids = stored['source_detection_index'].tolist()
            if seen.intersection(ids):
                raise ValueError('Duplicate detection IDs')
            seen.update(ids)
            if mode == 'top1_alpha1':
                joints = torch.as_tensor(stored['top1_joint'], device=args.device)
                np.testing.assert_array_equal(stored['uncertainty'].argmax(1), stored['top1_joint'])
                coords, scores = candidate_from_cache(base.head, stored, joints,
                                                      torch.ones(len(ids), device=args.device), args.device)
                arrays.update(top1_coords=coords, top1_scores=scores)
            elif mode == 'original':
                batch = next(loader)
                processed = base.data_preprocessor(batch, training=False)
                samples = processed['data_samples']
                np.testing.assert_array_equal([int(s.id) for s in samples], ids)
                np.testing.assert_array_equal([int(s.img_id) for s in samples], stored['image_id'])
                np.testing.assert_array_equal(np.stack([s.metainfo['input_scale'] for s in samples]), stored['input_scale'])
                feats = base.extract_feat(processed['inputs'])
                predictions = base.add_pred_to_datasample(
                    base.head.predict(feats, samples, base.test_cfg), None, deepcopy(samples))
                coords = np.stack([array(p.pred_instances.keypoints)[0, :, :2] for p in predictions])
                scores = np.stack([array(p.pred_instances.keypoint_scores)[0] for p in predictions])
                arrays.update(original_coords=coords, original_scores=scores)
            if mode != 'p0':
                prefix = 'top1' if mode == 'top1_alpha1' else 'original'
                dc = np.abs(coords - stored[prefix + '_coords']).max(axis=(1, 2))
                ds = np.abs(scores - stored[prefix + '_scores']).max(axis=1)
                delta['changed_instances'] += int(np.count_nonzero((dc != 0) | (ds != 0)))
                delta['max_coordinate_difference'] = max(delta['max_coordinate_difference'], float(dc.max()))
                delta['max_score_difference'] = max(delta['max_score_difference'], float(ds.max()))
            rows = list(records_from_arrays(cache, arrays, mode))
            cpu.add_records(rows)
            for row in rows:
                if count:
                    raw.write(',\n')
                json.dump(row, raw, allow_nan=False)
                count += 1
                official.process([], [dict(id=row['source_detection_index'], img_id=row['image_id'],
                    pred_instances=dict(keypoints=np.asarray([row['keypoints']], np.float32),
                        keypoint_scores=np.asarray([row['keypoint_scores']], np.float32),
                        bboxes=np.asarray([row['bbox']], np.float32),
                        bbox_scores=np.asarray([row['bbox_score']], np.float32)),
                    gt_instances=dict(bbox_scales=np.asarray([row['input_scale']], np.float32)))])
            if step == 1 or step % 100 == 0 or step == len(paths):
                atomic_json(args.output / 'progress.json', dict(mode=mode, stage='inference',
                    count=count, batches=step, total_batches=len(paths), seconds=time.monotonic()-started))
                print(f'{mode}: {count:,} detections, batch {step}/{len(paths)}', flush=True)
        raw.write(']\n')
    if not args.limit_batches and seen != set(cache.expected):
        raise ValueError('Missing detections')
    if loader is not None and not args.limit_batches:
        assert next(loader, None) is None
    if args.limit_batches:
        result = dict(sample_only=True, count=count, comparison_to_old_cache=delta,
                      note='Smoke check only; no formal AP/AR produced')
    else:
        print(mode + ': evaluating all 5,000 images', flush=True)
        atomic_json(args.output / 'progress.json', dict(mode=mode, stage='evaluation',count=count))
        with (folder / 'evaluation.log').open('a') as log, contextlib.redirect_stdout(log):
            reference = dict(official.evaluate(count))
            metrics, audit = cpu.evaluate()
            final_records = json.loads((folder / 'predictions.keypoints.json').read_text())
            final, final_audit = cpu.evaluate_coco_results(final_records)
        assert set(reference) == set(metrics) == set(final)
        for key in reference:
            np.testing.assert_allclose([metrics[key],final[key]], reference[key], rtol=0, atol=1e-12)
        assert audit['evaluation_image_count'] == 5000 and audit['num_input_instances'] == 104125
        assert len(cpu.last_coco_results) == len(final_records)
        for a, b in zip(cpu.last_coco_results, final_records):
            assert a['image_id'] == b['image_id']
            np.testing.assert_array_equal(a['keypoints'], b['keypoints'])
            np.testing.assert_array_equal(a['bbox'], b['bbox'])
            np.testing.assert_allclose(a['score'], b['score'], rtol=0, atol=1e-12)
        result = dict(sample_only=False, metrics=metrics, reference_metrics=reference,
                      final_json_metrics=final, audit=audit, final_audit=final_audit,
                      all_10_metrics_match=True, postprocessing_records_match=True,
                      comparison_to_old_cache=delta,
                      raw_predictions_sha256=hash_file(folder / 'raw_predictions_before_nms.json'),
                      final_predictions_sha256=hash_file(folder / 'predictions.keypoints.json'))
        print(f'{mode}: AP {metrics["coco/AP"]*100:.4f}, AR {metrics["coco/AR"]*100:.4f}', flush=True)
    result.update(mode=mode, cache_fingerprint=cache.fingerprint,
                  annotation_sha256=hash_file(args.ann), checkpoint_sha256=hash_file(args.checkpoint),
                  precision=runtime_record(), elapsed_seconds=time.monotonic()-started,
                  training=False, optimizer_steps=0,
                  inference_source='fixed_cached_P0' if mode=='p0' else ('cached_features' if mode=='top1_alpha1' else 'original_images'))
    atomic_json(folder / 'metrics.json', result)
    del cpu, official
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit-batches', type=int, default=0, help='Smoke only; no AP/AR')
    p.add_argument('--modes', nargs='+', choices=['p0','top1_alpha1','original'], default=['p0','top1_alpha1','original'])
    args = p.parse_args()
    if args.limit_batches < 0 or args.workers < 0 or len(set(args.modes)) != len(args.modes):
        p.error('Invalid limit/workers or duplicate modes')
    configure_legacy_precision()
    args.ann = args.data / 'annotations/person_keypoints_val2017.json'
    assert hash_file(args.checkpoint) == EXPECTED_CHECKPOINT
    cache = ReferenceCache(args.cache)
    cache.verify(True)
    assert len(cache.expected) == 104125
    protocol = cache.run['protocol']
    assert protocol['dataset_split'] == 'val2017' and protocol['sample_only'] is False
    assert protocol['flip_test'] is False
    assert hash_file(args.ann) == protocol['source_annotation_sha256']
    detection = args.data / 'person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
    assert hash_file(detection) == protocol['source_detection_sha256']
    metadata, records = load_detection_instances(detection,args.ann,args.data/'val2017')
    assert {r['source_detection_index']:r['image_id'] for r in records} == cache.expected
    assert len(json.loads(args.ann.read_text())['images']) == 5000
    for relative, digest in protocol['code_hashes'].items():
        # Adapters gained explicit precision initialization; model implementations did not change.
        if relative.startswith('mmpose/mmpose/'):
            assert hash_file(ROOT / relative) == digest
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'model_loading.log').open('w') as log, contextlib.redirect_stdout(log):
        base, cfg, meta = load_base(args.checkpoint,args.device)
    from mmengine.logging import MMLogger
    MMLogger.get_current_instance()
    atomic_json(args.output/'provenance.json', dict(precision=runtime_record(),
        cache_fingerprint=cache.fingerprint, detection_sha256=hash_file(detection),
        annotation_sha256=hash_file(args.ann), checkpoint_sha256=hash_file(args.checkpoint),
        source_hashes={n:hash_file(Path(__file__).with_name(n)) for n in
            ['rerun_baselines.py','precision_policy.py','build_shared_cache.py','build_detection_cache.py',
             'eval_unified.py','evaluate_detection_cache.py']},
        sample_only=bool(args.limit_batches),training=False))
    summary = {}
    for mode in args.modes:
        summary[mode] = run_mode(mode,args,cache,base,cfg,meta,records)
        atomic_json(args.output / 'baseline_summary.json',summary)
    atomic_json(args.output / 'progress.json',dict(stage='complete', modes=args.modes,training=False))
    print('COMPLETE: inference/evaluation only, no training',flush=True)


if __name__ == '__main__':
    main()
