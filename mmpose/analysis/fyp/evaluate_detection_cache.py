"""Export and independently evaluate frozen detection-cache baselines.

Run in cigpose once for official CocoMetric reference and CPU pre-NMS/final
JSON cross-checks. No GPU forward is repeated. M's eval_unified.py remains
the CPU-only standalone entry for delivered JSON files.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from detection_cache import ReferenceCache
from shared_cache import atomic_json
from eval_unified import UnifiedPoseEvaluator


def records_from_arrays(cache, a, mode):
    prefix = {'original': 'original', 'p0': 'P_0', 'top1_alpha1': 'top1'}[mode]
    for i, idx in enumerate(a['source_detection_index']):
        yield dict(image_id=int(a['image_id'][i]),
                   detection_instance_id=cache.instance_id(idx), source_detection_index=int(idx),
                   category_id=1, bbox=a['bbox'][i].tolist(), bbox_format='xyxy',
                   bbox_score=float(a['bbox_score'][i]), input_scale=a['input_scale'][i].tolist(),
                   keypoints=a[prefix+'_coords'][i].tolist(),
                   keypoint_scores=a[prefix+'_scores'][i].tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--ann', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.cache, args.ann, args.output = [x.resolve() for x in (args.cache, args.ann, args.output)]
    args.output.mkdir(parents=True, exist_ok=True)
    cache = ReferenceCache(args.cache)
    cache.verify(True)
    sys.path.insert(0, str(ROOT / 'mmpose'))
    os.chdir(ROOT / 'mmpose')
    from mmpose.evaluation.metrics import CocoMetric
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    meta = parse_pose_metainfo(dict(from_file='configs/_base_/datasets/coco.py'))
    from mmengine.logging import MMLogger
    MMLogger.get_current_instance()  # bind logger before redirecting per-mode stdout
    summary = {}
    for mode in ('original', 'p0', 'top1_alpha1'):
        folder = args.output / mode
        folder.mkdir(exist_ok=True)
        metric = CocoMetric(ann_file=str(args.ann), outfile_prefix=str(folder/'predictions'),
                            score_mode='bbox_keypoint', keypoint_score_thr=.2, nms_mode='oks_nms', nms_thr=.9)
        metric.dataset_meta = meta
        cpu = UnifiedPoseEvaluator(args.ann)
        n = 0
        with (folder/'raw_predictions_before_nms.json').open('w') as out:
            out.write('[')
            for path in cache.shards():
                a = cache.load_shard(path)
                rows = list(records_from_arrays(cache, a, mode))
                for r in rows:
                    if n: out.write(',\n')
                    json.dump(r, out, allow_nan=False)
                    n += 1
                    metric.process([], [dict(id=r['source_detection_index'], img_id=r['image_id'],
                        pred_instances=dict(keypoints=np.asarray([r['keypoints']], np.float32),
                            keypoint_scores=np.asarray([r['keypoint_scores']], np.float32),
                            bboxes=np.asarray([r['bbox']], np.float32),
                            bbox_scores=np.asarray([r['bbox_score']], np.float32)),
                        gt_instances=dict(bbox_scales=np.asarray([r['input_scale']], np.float32)))])
                cpu.add_records(rows)
            out.write(']\n')
        with (folder/'evaluation.log').open('w') as log, contextlib.redirect_stdout(log):
            official = dict(metric.evaluate(n))
            pre, audit = cpu.evaluate()
            final_records = json.loads((folder/'predictions.keypoints.json').read_text())
            final, final_audit = cpu.evaluate_coco_results(final_records)
        np.testing.assert_allclose(list(official.values()), list(pre.values()), rtol=0, atol=1e-12)
        np.testing.assert_allclose(list(official.values()), list(final.values()), rtol=0, atol=1e-12)
        # Check post-processing per output record, not just rounded AP.
        assert len(cpu.last_coco_results) == len(final_records)
        for a, b in zip(cpu.last_coco_results, final_records):
            assert a['image_id'] == b['image_id']
            np.testing.assert_array_equal(a['keypoints'], b['keypoints'])
            np.testing.assert_allclose(a['score'], b['score'], rtol=0, atol=1e-12)
        assert audit['evaluation_image_count'] == 5000
        result = dict(reference_metrics=official, cpu_pre_nms_metrics=pre, cpu_final_metrics=final,
                      audit=audit, all_10_metrics_match=True, postprocessing_records_match=True,
                      input_count=n, box_protocol='detection', flip_test=False)
        atomic_json(folder/'metrics.json', result)
        summary[mode] = result
        print(mode, official, flush=True)
        del metric, cpu
    eq = dict(count=0, joint_mismatch_count=0, prediction_difference_count=0,
              max_coordinate_abs_difference=0., max_score_abs_difference=0.)
    for path in cache.shards():
        a = cache.load_shard(path)
        d = np.max(np.abs(a['original_coords']-a['top1_coords']), axis=(1,2))
        ds = np.max(np.abs(a['original_scores']-a['top1_scores']), axis=1)
        eq['count'] += len(d)
        eq['joint_mismatch_count'] += int(np.sum(a['original_joint'] != a['top1_joint']))
        eq['prediction_difference_count'] += int(np.sum((d != 0) | (ds != 0)))
        eq['max_coordinate_abs_difference'] = max(eq['max_coordinate_abs_difference'], float(d.max()))
        eq['max_score_abs_difference'] = max(eq['max_score_abs_difference'], float(ds.max()))
    eq['exact_prediction_equivalence'] = eq['prediction_difference_count'] == 0
    atomic_json(args.output/'equivalence_report.json', eq)
    atomic_json(args.output/'baseline_summary.json', summary)


if __name__ == '__main__':
    main()
