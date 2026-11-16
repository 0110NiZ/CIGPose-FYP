"""Fixed detection inputs -> resumable reference cache and three baseline paths.

No GT annotations are passed into model inputs. Source index plus the full
detection SHA identifies an instance; it is never a GT annotation ID.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from read_val_detection_boxes import load_detection_instances, select_samples
from detection_cache import ReferenceCache, CacheWriter, SHAPES, INTEGER_FIELDS
from shared_cache import hash_file, atomic_json
from build_shared_cache import array, decode_original, check_reloaded, EXPECTED_CHECKPOINT


class DetectionDataset:
    def __init__(self, records, pipeline, metainfo):
        from mmengine.dataset import Compose
        self.records, self.pipeline, self.metainfo = records, Compose(pipeline), metainfo

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        item = dict(img_id=r['image_id'], id=r['source_detection_index'],
                    img_path=r['image_path'], category_id=1,
                    bbox=np.asarray([r['bbox_xyxy']], dtype=np.float32),
                    bbox_score=np.asarray([r['bbox_score']], dtype=np.float32))
        for k in ('flip_indices', 'upper_body_ids', 'lower_body_ids',
                  'flip_pairs', 'dataset_keypoint_weights', 'skeleton_links'):
            if k in self.metainfo:
                item[k] = deepcopy(self.metainfo[k])
        return self.pipeline(item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--ann', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.batch_size != 32:
        p.error('The shared legacy precision profile requires batch-size=32')
    for k in ('detections', 'ann', 'images', 'checkpoint', 'output'):
        setattr(args, k, getattr(args, k).resolve())
    metadata, records = load_detection_instances(args.detections, args.ann, args.images)
    if args.smoke:
        records = select_samples(records)
    assert hash_file(args.checkpoint) == EXPECTED_CHECKPOINT
    sys.path.insert(0, str(ROOT / 'mmpose'))
    os.chdir(ROOT / 'mmpose')
    import torch
    from torch.utils.data import DataLoader, Subset
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmpose.apis import init_model
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    from mmpose.utils import register_all_modules
    register_all_modules()
    from precision_policy import configure_legacy_precision
    precision_record = configure_legacy_precision()
    cfg_path = ROOT / 'mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_coco-256x192_detection-single-view.py'
    cfg = Config.fromfile(str(cfg_path))
    cfg.test_dataloader.dataset.bbox_file = str(args.detections)
    cfg.test_dataloader.dataset.ann_file = str(args.ann)
    cfg.test_dataloader.dataset.data_prefix.img = str(args.images) + '/'
    cfg.test_evaluator.ann_file = str(args.ann)
    assert cfg.model.test_cfg.flip_test is False
    meta = parse_pose_metainfo(dict(from_file='configs/_base_/datasets/coco.py'))
    dataset = DetectionDataset(records, cfg.val_pipeline, meta)
    codepaths = [Path(__file__), Path(__file__).with_name('detection_cache.py'),
                 ROOT / 'scripts/read_val_detection_boxes.py',
                 ROOT / 'mmpose/mmpose/models/heads/coord_cls_heads/cig_head.py',
                 ROOT / 'mmpose/mmpose/models/pose_estimators/topdown.py']
    protocol = dict(metadata, checkpoint_sha256=EXPECTED_CHECKPOINT, flip_test=False,
                    flip_protocol='single_view', precision='float32', seed=42,
                    bbox_format='xyxy', source_bbox_format='xywh', base_model_eval=True,
                    code_hashes={str(f.relative_to(ROOT)): hash_file(f) for f in codepaths},
                    resolved_config=cfg.pretty_text, sample_only=args.smoke,
                    batch_size=args.batch_size)
    protocol.update(precision_profile=precision_record['precision_profile'],
                    precision_profile_sha256=precision_record['precision_profile_sha256'])
    cache = ReferenceCache.create(args.output, protocol,
              [(r['image_id'], r['source_detection_index']) for r in records], args.resume)
    cfg.dump(str(args.output / 'resolved_config.py'))
    atomic_json(args.output / 'detection_manifest.json', dict(metadata,
                detection_instance_ids=[r['detection_instance_id'] for r in records]))
    model = init_model(cfg, str(args.checkpoint), device='cuda:0')
    model.requires_grad_(False).eval()
    expected = {r['source_detection_index']: r for r in records}
    start = time.time()
    with CacheWriter(cache) as writer, torch.inference_mode():
        remaining = [i for i, r in enumerate(records) if r['source_detection_index'] not in writer.seen]
        loader = DataLoader(Subset(dataset, remaining), batch_size=args.batch_size,
                            num_workers=args.workers, collate_fn=pseudo_collate, shuffle=False)
        for step, batch in enumerate(loader):
            processed = model.data_preprocessor(batch, training=False)
            samples = processed['data_samples']
            feats = model.extract_feat(processed['inputs'])
            original = model.add_pred_to_datasample(
                model.head.predict(feats, samples, model.test_cfg), None, deepcopy(samples))
            state = model.head.extract_refinement_state(feats)
            original_joint = torch.topk(state['uncertainty'], 1, dim=1).indices[:, 0]
            p0 = decode_original(model, state['p0_simcc'], samples)
            modified, _ = model.head.cim.apply_intervention(state['features'], state['top1_indices'], torch.ones(len(samples), device='cuda:0'))
            top1 = decode_original(model, model.head.predict_from_joint_features(modified), samples)
            cpu = {k: array(state[k]) for k in ('features', 'uncertainty', 'confidence', 'top1_indices')}
            rows = {k: [] for k in SHAPES}
            for i, result in enumerate(p0):
                idx = int(result.id)
                r = expected[idx]
                assert int(result.img_id) == r['image_id']
                pred = result.pred_instances
                box, score = array(pred.bboxes)[0], array(pred.bbox_scores).reshape(-1)[0]
                np.testing.assert_array_equal(box, np.asarray(r['bbox_xyxy'], np.float32))
                assert score == np.float32(r['bbox_score'])
                v = dict(features=cpu['features'][i], uncertainty=cpu['uncertainty'][i],
                         confidence=cpu['confidence'][i], top1_joint=cpu['top1_indices'][i],
                         original_joint=int(original_joint[i]), image_id=r['image_id'],
                         source_detection_index=idx, bbox=box, bbox_score=score,
                         original_size=[r['image_width'], r['image_height']])
                for prefix, results in [('P_0', p0), ('original', original), ('top1', top1)]:
                    v[prefix+'_coords'] = array(results[i].pred_instances.keypoints)[0, :, :2]
                    v[prefix+'_scores'] = array(results[i].pred_instances.keypoint_scores)[0]
                for k in ('input_center', 'input_scale', 'input_size'):
                    v[k] = result.metainfo[k]
                for k in rows:
                    rows[k].append(v[k])
            arrays = {k: np.asarray(v, dtype=np.int64 if k in INTEGER_FIELDS else np.float32)
                      for k, v in rows.items()}
            writer.append(arrays)
            if step == 0:
                loaded = cache.load_shard(cache.directory / f'shard-{int(arrays["source_detection_index"][0]):012d}.npz')
                check_reloaded(model, loaded, samples, state, 'cuda:0')
                # Full replacement and candidate independence, including per-person actions.
                before = state['features'].clone()
                joints = torch.arange(len(samples), device='cuda:0') % 17
                replaced, _ = model.head.cim.apply_intervention(before, joints, torch.ones(len(samples), device='cuda:0'))
                zero, _ = model.head.cim.apply_intervention(before, joints, torch.zeros(len(samples), device='cuda:0'))
                torch.testing.assert_close(zero, before, rtol=0, atol=0)
                torch.testing.assert_close(state['features'], before, rtol=0, atol=0)
                again, _ = model.head.cim.apply_intervention(before, joints, torch.ones(len(samples), device='cuda:0'))
                torch.testing.assert_close(replaced, again, rtol=0, atol=0)
                for b, j in enumerate(joints):
                    torch.testing.assert_close(replaced[b,j], model.head.cim.canonical_features.weight[j], rtol=0, atol=0)
                atomic_json(args.output / 'forward_checks.json', dict(
                    passed=True, checked_count=len(samples), alpha0_and_cache_reload=True,
                    mixed_joint_alpha_cache_vs_live=True, candidate_independence=True,
                    source_id_bbox_score_alignment=True, image_decoding=True))
            if step % 16 == 0:
                print(f'Cached {len(writer.seen)}/{len(records)}; elapsed {time.time()-start:.1f}s', flush=True)
    seen = cache.verify(require_complete=True)
    atomic_json(args.output / 'completion.json', dict(passed=True, count=len(seen),
                images=len({r['image_id'] for r in records}), elapsed_seconds=time.time()-start,
                device=torch.cuda.get_device_name(), torch_version=torch.__version__))
    print('COMPLETE', len(seen), flush=True)


if __name__ == '__main__':
    main()
