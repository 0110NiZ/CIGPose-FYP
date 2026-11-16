"""A-only validation: select one joint, alpha=1, always accept; no GT matching."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from a_contract import model_inputs
from a_runtime import (ROOT, load_a_checkpoint, load_frozen_base, require_val_cache, setup)
from build_shared_cache import candidate_from_cache
from detection_cache import ReferenceCache
from gain_predictor import select_joint, to_device
from shared_cache import atomic_json, hash_file
from precision_policy import assert_legacy_precision, runtime_record

sys.path.insert(0, str(ROOT / 'scripts'))


@torch.inference_mode()
def predict_batch(predictor, base, arrays, device):
    assert_legacy_precision()
    if predictor.training or base.training or any(p.requires_grad for p in base.parameters()):
        raise ValueError('Inference requires eval A and frozen eval CIGPose')
    gains = predictor(**to_device(model_inputs(arrays), device))
    joints = select_joint(gains)
    coords, scores = candidate_from_cache(base.head, arrays, joints,
                                          torch.ones(len(joints), device=device), device)
    if not np.isfinite(coords).all() or not np.isfinite(scores).all():
        raise ValueError('Nonfinite candidate output')
    return gains.cpu().numpy(), joints.cpu().numpy(), coords, scores


def prediction_records(cache, arrays, coords, scores):
    for i, idx in enumerate(arrays['source_detection_index']):
        yield dict(image_id=int(arrays['image_id'][i]),
                   detection_instance_id=cache.instance_id(idx), source_detection_index=int(idx),
                   category_id=1, bbox=arrays['bbox'][i].tolist(), bbox_format='xyxy',
                   bbox_score=float(arrays['bbox_score'][i]), input_scale=arrays['input_scale'][i].tolist(),
                   keypoints=coords[i].tolist(), keypoint_scores=scores[i].tolist())


def validate_a(predictor, base, cache, ann, output, device, provenance=None):
    """Full val only. The official public CPU evaluator applies scoring/NMS once."""
    from eval_unified import UnifiedPoseEvaluator
    require_val_cache(cache)
    ann, output = Path(ann).resolve(), Path(output).resolve()
    ann_data = json.loads(ann.read_text())
    expected_ann_sha = cache.run['protocol'].get('source_annotation_sha256')
    # Detector metadata versions may name the annotation hash differently.
    if expected_ann_sha and hash_file(ann) != expected_ann_sha:
        raise ValueError('Validation annotation differs from cached provenance')
    image_ids = {int(x['id']) for x in ann_data['images']}
    if len(image_ids) != 5000 or not set(cache.expected.values()).issubset(image_ids):
        raise ValueError('Use the full original val2017 annotations (5000 images)')
    if output.exists():
        raise FileExistsError('Validation output must be new: ' + str(output))
    output.mkdir(parents=True)
    predictor.eval()
    evaluator = UnifiedPoseEvaluator(ann)
    count, seen, histogram = 0, set(), np.zeros(17, np.int64)
    raw_path = output / 'raw_predictions_before_nms.json'
    raw_tmp = raw_path.with_suffix('.json.tmp')
    choice_path = output / 'selections.jsonl'
    choice_tmp = choice_path.with_suffix('.jsonl.tmp')
    with raw_tmp.open('w') as raw, choice_tmp.open('w') as choices:
        raw.write('[')
        for path in cache.shards():
            arrays = cache.load_shard(path)
            ids = arrays['source_detection_index'].tolist()
            if seen.intersection(ids):
                raise ValueError('Duplicate validation detection')
            seen.update(ids)
            gains, joints, coords, scores = predict_batch(predictor, base, arrays, device)
            rows = list(prediction_records(cache, arrays, coords, scores))
            evaluator.add_records(rows)
            histogram += np.bincount(joints, minlength=17)
            for i, row in enumerate(rows):
                if count:
                    raw.write(',\n')
                json.dump(row, raw, allow_nan=False)
                json.dump(dict(image_id=row['image_id'], detection_instance_id=row['detection_instance_id'],
                               selected_joint=int(joints[i]), predicted_gain=gains[i].tolist()),
                          choices, allow_nan=False)
                choices.write('\n')
                count += 1
        raw.write(']\n')
    if seen != set(cache.expected):
        raise ValueError('Validation did not cover every supplied detection')
    raw_tmp.replace(raw_path)
    choice_tmp.replace(choice_path)
    with (output / 'evaluation.log').open('w') as log, contextlib.redirect_stdout(log):
        metrics, audit = evaluator.evaluate()
    if audit['evaluation_image_count'] != 5000 or audit['num_input_instances'] != 104125:
        raise ValueError('Incorrect full validation coverage')
    atomic_json(output / 'predictions.keypoints.json', evaluator.last_coco_results)
    result = dict(metrics=metrics, audit=audit, method='A-only', alpha=1.0,
                  precision=runtime_record(),
                  always_accept=True, flip_test=False, cache_fingerprint=cache.fingerprint,
                  annotation_sha256=hash_file(ann), selection_counts=histogram.tolist(),
                  provenance=provenance or {})
    atomic_json(output / 'metrics.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--a-checkpoint', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--ann', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    setup()
    cache = ReferenceCache(args.cache)
    require_val_cache(cache)
    cache.verify(True)
    predictor, payload = load_a_checkpoint(args.a_checkpoint, args.device)
    val_images = {int(x['id']) for x in json.loads(args.ann.read_text())['images']}
    if set(payload['train_image_ids']).intersection(val_images):
        raise ValueError('Train/validation image overlap')
    base = load_frozen_base(args.base_checkpoint, args.device)
    result = validate_a(predictor, base, cache, args.ann, args.output, args.device,
                        dict(a_checkpoint_sha256=hash_file(args.a_checkpoint), epoch=payload['epoch']))
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
