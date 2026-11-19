"""Full fixed-detection val2017 AP used for epoch selection."""
import contextlib
import json
from pathlib import Path
import numpy as np

from Module_B.common import sha, digest, atomic_json
from Module_B.config import ALPHAS, EXPECTED_DETECTIONS
from Module_B.data import make_features

PROTOCOL = 'train2017_all_val2017_ap'
TRAIN_FILES = ('train.py', 'validate.py', 'evaluate.py', 'common.py',
               'config.py', 'data.py', 'model.py')


def improves(ap, best_ap):
    """Use unrounded AP; exact ties retain the earlier epoch."""
    if not np.isfinite(ap) or not 0 <= ap <= 1:
        raise ValueError('Validation AP must be finite and in [0,1]')
    return ap > best_ap


def check_history(history, train_people):
    if not history:
        raise ValueError('Empty validation history')
    best = None
    for epoch, row in enumerate(history, 1):
        validation = row['validation']
        ap = validation['validation_AP']
        improves(ap, -1.)
        if (row['epoch'] != epoch or row['train_people'] != train_people
                or validation['epoch'] != epoch
                or validation['detections'] != EXPECTED_DETECTIONS
                or validation['evaluation_image_count'] != 5000
                or len(validation['metrics']) != 10
                or not np.isfinite(row['train_scaled_huber_loss'])):
            raise ValueError('Incomplete epoch training/validation coverage')
        np.testing.assert_allclose(validation['metrics']['coco/AP'], ap, rtol=0, atol=1e-12)
        if best is None or improves(ap, best['validation']['validation_AP']):
            best = row
    return best


class ValEvaluator:
    def __init__(self, repo, cache_dir, ann, checkpoint, train_images, device):
        import torch
        from detection_cache import ReferenceCache
        from mmengine.config import Config
        from mmpose.apis import init_model
        from mmpose.datasets.datasets.utils import parse_pose_metainfo
        from Module_B.evaluate import validate_detection_protocol

        self.device = device
        self.cache = ReferenceCache(cache_dir)
        self.ann = Path(ann)
        self.ann_sha = sha(self.ann)
        validate_detection_protocol(self.cache, sha(checkpoint))
        cp = self.cache.run['protocol']
        if cp.get('annotation_sha256', self.ann_sha) != self.ann_sha:
            raise ValueError('Validation annotation/cache mismatch')
        meta = json.loads(self.ann.read_text(encoding='utf-8-sig'))
        self.images = {int(x['id']) for x in meta['images']}
        if len(self.images) != 5000 or not set(self.cache.expected.values()) <= self.images:
            raise ValueError('Validation must use all 5000 COCO val2017 images')
        if set(map(int, train_images)) & self.images:
            raise ValueError('Training and val2017 image identities overlap')
        del meta
        self.cache.verify(True)
        self.sources = self.cache.shards()
        self.source_info = [dict(name=p.name, sha256=sha(p)) for p in self.sources]
        self.run_sha = sha(self.cache.directory/'run.json')
        mmroot = Path(repo)/'mmpose'
        cfg_path = mmroot/'projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py'
        cfg = Config.fromfile(str(cfg_path))
        cfg.model.test_cfg['flip_test'] = False
        if cp.get('config_sha256', digest(cfg.to_dict())) != digest(cfg.to_dict()):
            raise ValueError('Config differs from frozen detection cache')
        self.pose = init_model(cfg, str(checkpoint), device=device)
        self.pose.requires_grad_(False).eval()
        self.dataset_meta = parse_pose_metainfo(dict(from_file='configs/_base_/datasets/coco.py'))
        files = [cfg_path, mmroot/'analysis/fyp/detection_cache.py',
                 mmroot/'analysis/fyp/build_shared_cache.py', mmroot/'analysis/fyp/shared_cache.py',
                 mmroot/'mmpose/models/heads/coord_cls_heads/cig_head.py',
                 mmroot/'mmpose/evaluation/metrics/coco_metric.py']
        self.provenance = dict(cache=str(self.cache.directory), cache_run_sha256=self.run_sha,
            annotation_sha256=self.ann_sha, source_shards=self.source_info,
            source_protocol=cp, checkpoint_sha256=sha(checkpoint),
            config_sha256=digest(cfg.to_dict()), implementation_sha256={str(p):sha(p) for p in files},
            expected_instances=EXPECTED_DETECTIONS, validation_image_count=5000,
            score_mode='bbox_keypoint', keypoint_score_thr=.2, nms_mode='oks_nms', nms_thr=.9,
            flip_test=False, gt_used_for_actions=False, gt_used_for_scoring=True)
        print(f'Full validation cache ready: {len(self.cache.expected)} detections, 5000 images', flush=True)

    def evaluate(self, model, mean, std, gain_scale, output, epoch, verify=False):
        import torch
        from mmpose.evaluation.metrics import CocoMetric
        from mmengine.logging import MMLogger
        from build_shared_cache import candidate_from_cache
        from Module_B.evaluate import records

        output = Path(output)
        scratch = output/'.validation'
        scratch.mkdir(exist_ok=True)
        report_name = 'best_recheck' if verify else f'epoch_{epoch:03d}'
        log_path = output/'epochs'/(report_name+'.log')
        if sha(self.cache.directory/'run.json') != self.run_sha or sha(self.ann) != self.ann_sha:
            raise ValueError('Validation cache metadata/annotation changed')
        MMLogger.get_current_instance()
        metric = CocoMetric(ann_file=str(self.ann), outfile_prefix=str(scratch/'predictions'),
            score_mode='bbox_keypoint', keypoint_score_thr=.2, nms_mode='oks_nms', nms_thr=.9)
        metric.dataset_meta = self.dataset_meta
        if set(map(int, metric.coco.getImgIds())) != self.images:
            raise ValueError('Official evaluator image set changed')
        was_training = model.training
        model.eval()
        seen = set()
        counts = {str(a):0 for a in ALPHAS}
        try:
            with torch.inference_mode():
                for seq, (path, info) in enumerate(zip(self.sources, self.source_info), 1):
                    if sha(path) != info['sha256']:
                        raise ValueError('Detection source changed during training: '+str(path))
                    a = self.cache.load_shard(path)
                    ids = set(map(int, a['source_detection_index']))
                    if seen & ids:
                        raise ValueError('Duplicate validation detection identity')
                    seen.update(ids)
                    features = (make_features(a)-mean)/std
                    if not np.isfinite(features).all():
                        raise ValueError('Non-finite validation input')
                    prediction = model(torch.from_numpy(features).to(self.device)).cpu().numpy()/gain_scale
                    if prediction.shape != (len(ids),4) or not np.isfinite(prediction).all():
                        raise ValueError('Invalid B validation output')
                    chosen = np.asarray(ALPHAS, np.float32)[prediction.argmax(axis=1)]
                    joints = torch.as_tensor(a['top1_joint'], device=self.device)
                    original = a['features'].copy()
                    coords, scores = candidate_from_cache(self.pose.head, a, joints,
                        torch.as_tensor(chosen, device=self.device), self.device)
                    np.testing.assert_array_equal(a['features'], original)
                    if coords.shape != (len(ids),17,2) or scores.shape != (len(ids),17):
                        raise ValueError('Invalid validation pose shape')
                    if not np.isfinite(coords).all() or not np.isfinite(scores).all():
                        raise ValueError('Non-finite validation pose')
                    if seq == 1:
                        again = model(torch.from_numpy(features).to(self.device)).cpu().numpy()/gain_scale
                        np.testing.assert_array_equal(prediction, again)
                    for r in records(self.cache, a, coords, scores):
                        metric.process([], [dict(id=r['source_detection_index'], img_id=r['image_id'],
                            pred_instances=dict(keypoints=np.asarray([r['keypoints']],np.float32),
                                keypoint_scores=np.asarray([r['keypoint_scores']],np.float32),
                                bboxes=np.asarray([r['bbox']],np.float32),
                                bbox_scores=np.asarray([r['bbox_score']],np.float32)),
                            gt_instances=dict(bbox_scales=np.asarray([r['input_scale']],np.float32)))])
                    for alpha in ALPHAS:
                        counts[str(alpha)] += int((chosen == alpha).sum())
                    if seq % 512 == 0 or seq == len(self.sources):
                        print(f'  val epoch {epoch}: {seq}/{len(self.sources)} shards', flush=True)
            if seen != set(self.cache.expected) or len(seen) != EXPECTED_DETECTIONS:
                raise ValueError('Incomplete full validation coverage')
            with log_path.open('w') as log, contextlib.redirect_stdout(log):
                metrics = {k:float(v) for k,v in metric.evaluate(len(seen)).items()}
            ap = metrics['coco/AP']
            improves(ap, -1.)
            if len(metrics) != 10 or not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError('Invalid official COCO metric results')
            prediction_file = scratch/'predictions.keypoints.json'
            report = dict(epoch=epoch, metrics=metrics, validation_AP=ap,
                detections=len(seen), evaluation_image_count=5000, bbox_protocol='fixed_detection',
                selected_alpha_counts=counts, flip_test=False,
                predictions_sha256=sha(prediction_file), source_cache_run_sha256=self.run_sha,
                official_coco_metric=True, independent_test=False)
            atomic_json(output/'epochs'/(report_name+'.json'), report)
            return report, prediction_file
        finally:
            model.train(was_training)
