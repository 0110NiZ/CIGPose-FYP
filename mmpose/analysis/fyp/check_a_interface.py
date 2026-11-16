"""N-side readiness check on real caches. No labels, training or AP claims."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from a_contract import label_metadata_template, model_inputs, require_train_cache
from a_runtime import (ROOT, load_frozen_base, require_cache_model, require_val_cache, setup)
from build_shared_cache import candidate_from_cache
from detection_cache import ReferenceCache as DetectionCache
from model import MLP as GainPredictor, to_device
from predict_a import predict_batch, prediction_records
from shared_cache import ReferenceCache, atomic_json, hash_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache', type=Path, required=True)
    p.add_argument('--val-cache', type=Path, required=True)
    p.add_argument('--val-ann', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    setup()
    if args.output.exists():
        raise FileExistsError('Choose a new check output directory')
    train, val = ReferenceCache(args.train_cache), DetectionCache(args.val_cache)
    require_train_cache(train)
    require_cache_model(train)
    require_val_cache(val)
    # Read-only full coverage verification; not sample-only acceptance.
    train.verify(True)
    val.verify(True)
    val_images = {int(x['id']) for x in json.loads(args.val_ann.read_text())['images']}
    if (hash_file(args.val_ann) != val.run['protocol']['source_annotation_sha256']
            or len(val_images) != 5000 or not set(val.expected.values()).issubset(val_images)):
        raise ValueError('Expected full original val2017 annotation with approved hash')
    if set(train.expected.values()).intersection(val_images):
        raise ValueError('Train/val image overlap')
    predictor = GainPredictor().to(args.device).eval()
    initial = {k: v.detach().clone() for k, v in predictor.state_dict().items()}
    base = load_frozen_base(args.base_checkpoint, args.device)
    from eval_unified import UnifiedPoseEvaluator
    evaluator = UnifiedPoseEvaluator(args.val_ann)
    cases = []
    for name, cache in [('train_gt', train), ('val_detection', val)]:
        paths = cache.shards()
        for idx in sorted({0, len(paths)//2, len(paths)-1}):
            arrays = cache.load_shard(paths[idx])
            # Use a complete original shard to retain the public numerical protocol.
            with torch.inference_mode():
                inputs = model_inputs(arrays)
                gains, joints, coords, scores = predict_batch(predictor, base, arrays, args.device)
                zero_coords, zero_scores = candidate_from_cache(
                    base.head, arrays, torch.as_tensor(joints, device=args.device),
                    torch.zeros(len(joints), device=args.device), args.device)
                np.testing.assert_allclose(zero_coords, arrays['P_0_coords'], atol=.1, rtol=0)
                np.testing.assert_allclose(zero_scores, arrays['P_0_scores'], atol=1e-5, rtol=0)
                if name == 'val_detection':
                    rows = list(prediction_records(cache, arrays, coords, scores))
                    if len({r['detection_instance_id'] for r in rows}) != len(rows):
                        raise ValueError('Export identity collision')
                    # The real evaluator's pre-NMS ingestion must accept these records.
                    # No GT scoring/metrics are performed in this check.
                    evaluator.add_records(rows)
                if gains.shape != (len(joints), 17) or not np.isfinite(gains).all():
                    raise ValueError('Invalid A output')
            cases.append(dict(split=name, shard=paths[idx].name, count=len(joints),
                              output_shape=list(gains.shape), input_keys=sorted(inputs),
                              alpha0_max_coordinate_error=float(np.max(np.abs(zero_coords-arrays['P_0_coords']))),
                              alpha0_max_score_error=float(np.max(np.abs(zero_scores-arrays['P_0_scores'])))))
    for k, v in predictor.state_dict().items():
        torch.testing.assert_close(v, initial[k], atol=0, rtol=0)
    if any(p.grad is not None for p in base.parameters()) or any(p.grad is not None for p in predictor.parameters()):
        raise ValueError('Readiness check unexpectedly computed gradients')
    args.output.mkdir(parents=True)
    atomic_json(args.output/'Y_metadata_template.json', label_metadata_template(train))
    atomic_json(args.output/'interface_check.json', dict(
        passed=True, trained=False, optimizer_steps=0, gain_labels_generated=0,
        random_model_for_interface_test_only=True, train_count=len(train.expected),
        val_detection_count=len(val.expected), parameter_count=sum(p.numel() for p in predictor.parameters()),
        device=args.device, gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        torch_version=torch.__version__, cases=cases,
        train_cache_fingerprint=train.fingerprint, val_cache_fingerprint=val.fingerprint))
    print('PASS: interfaces checked; no training, no gain labels, no learned checkpoint saved.')


if __name__ == '__main__':
    main()
