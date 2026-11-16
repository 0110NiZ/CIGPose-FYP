"""N's full-data MLP trainer. Default is validation-only preflight, NOT training.

Real optimization requires an explicit --run-training and Y's complete labels.
No label generation, base-model optimization or test-dev evaluation occurs here.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from a_contract import CHECKPOINT_SCHEMA, INPUT_SCHEMA
from a_dataset import ATrainingData, fit_input_statistics
from a_runtime import (EXPECTED_CHECKPOINT, load_frozen_base, require_cache_model,
                       require_val_cache, save_checkpoint, setup)
from detection_cache import ReferenceCache as DetectionCache
from gain_predictor import GainPredictor, to_device
from shared_cache import atomic_json, hash_file


def gain_loss(prediction, target, gain_scale=100.0):
    """Scaling both sides keeps signed gains and their relative magnitudes."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 17:
        raise ValueError('Loss requires [B,17] prediction and target')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite prediction/target')
    return F.smooth_l1_loss(prediction * gain_scale, target * gain_scale)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--val-cache', type=Path, required=True)
    p.add_argument('--val-ann', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-training', action='store_true', help='Explicitly enable optimizer updates')
    p.add_argument('--resume', action='store_true', help='Continue this output directory from last.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-shards', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--gain-scale', type=float, default=100.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if (min(args.epochs, args.eval_every, args.batch_size, args.buffer_shards) < 1
            or not np.isfinite([args.lr, args.weight_decay, args.gain_scale]).all()
            or args.lr <= 0 or args.weight_decay < 0 or args.gain_scale <= 0 or args.seed < 0):
        p.error('Invalid training settings')
    precision_record = setup(args.seed)
    data = ATrainingData(args.train_cache, args.labels)
    require_cache_model(data.cache)
    val = DetectionCache(args.val_cache)
    require_val_cache(val)
    val.verify(True)
    train_images = sorted(set(data.cache.expected.values()))
    if set(train_images).intersection(val.expected.values()):
        raise ValueError('Train/val image overlap')
    if hash_file(args.base_checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('Wrong CIGPose checkpoint')
    val_ann = json.loads(args.val_ann.read_text())
    if hash_file(args.val_ann) != val.run['protocol']['source_annotation_sha256']:
        raise ValueError('Validation annotation differs from approved cache provenance')
    val_images = {int(x['id']) for x in val_ann['images']}
    if len(val_images) != 5000 or not set(val.expected.values()).issubset(val_images):
        raise ValueError('Expected complete original val2017 annotation')
    if set(train_images).intersection(val_images):
        raise ValueError('Train overlaps validation images without detections')
    code_files = ['a_contract.py', 'a_dataset.py', 'gain_predictor.py', 'a_runtime.py',
                  'train_a.py', 'predict_a.py', 'eval_unified.py', 'build_shared_cache.py', 'precision_policy.py']
    settings = dict(
        precision_profile=precision_record['precision_profile'],
        precision_profile_sha256=precision_record['precision_profile_sha256'],
        epochs=args.epochs, eval_every=args.eval_every, batch_size=args.batch_size,
        buffer_shards=args.buffer_shards, lr=args.lr, weight_decay=args.weight_decay,
        gain_scale=args.gain_scale, seed=args.seed, device=args.device,
        train_cache_fingerprint=data.cache.fingerprint, label_fingerprint=data.labels.fingerprint,
        val_cache_fingerprint=val.fingerprint, val_annotation_sha256=hash_file(args.val_ann),
        base_checkpoint_sha256=EXPECTED_CHECKPOINT, input_schema=INPUT_SCHEMA,
        code_hashes={n: hash_file(Path(__file__).with_name(n)) for n in code_files},
        torch_version=torch.__version__, numpy_version=np.__version__)
    print(json.dumps(dict(preflight='PASS', train_count=len(data.cache.expected),
                          val_detection_count=len(val.expected),
                          optimization_enabled=args.run_training), indent=2), flush=True)
    if not args.run_training:
        print('No training performed. Add --run-training only when ready.')
        return

    output = args.output.resolve()
    model = GainPredictor().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_ap, history = 0, None, []
    if args.resume:
        prior = torch.load(output / 'last.pt', map_location='cpu')
        if (prior.get('schema_version') != CHECKPOINT_SCHEMA or prior.get('settings') != settings
                or prior.get('trained') is not True):
            raise ValueError('Resume settings, inputs, labels or code changed; use a new output directory')
        model.load_state_dict(prior['model_state'])
        optimizer.load_state_dict(prior['optimizer_state'])
        start_epoch, best_ap, history = prior['epoch'], prior['best_val_ap'], prior['history']
        torch.set_rng_state(prior['torch_rng_state'])
        if torch.cuda.is_available() and prior.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(prior['cuda_rng_state'])
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Use a new training output directory or --resume')
        output.mkdir(parents=True, exist_ok=True)
        mean, std = fit_input_statistics(data.cache)
        model.set_input_statistics(mean, std)
    if start_epoch >= args.epochs:
        print('Requested epochs already complete; no updates performed.')
        return
    # Only the A model is present in optimizer; base is used exclusively for validation.
    base = load_frozen_base(args.base_checkpoint, args.device)
    from predict_a import validate_a
    lock = output / '.training.lock'
    with lock.open('x') as f:
        import os
        f.write(str(os.getpid()))
    try:
        atomic_json(output / 'settings.json', settings)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            begin = time.monotonic()
            model.train()
            total, loss_sum, error_sum = 0, 0., 0.
            for inputs, target, ids in data.batches(args.batch_size, args.seed, epoch, args.buffer_shards):
                x = to_device(inputs, args.device)
                y = torch.as_tensor(target, dtype=torch.float32, device=args.device)
                prediction = model(**x)
                loss = gain_loss(prediction, y, args.gain_scale)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += len(ids)
                loss_sum += float(loss.detach()) * len(ids)
                error_sum += float((prediction.detach() - y).abs().mean()) * len(ids)
            if total != len(data.cache.expected):
                raise ValueError('Training epoch omitted or duplicated instances')
            row = dict(epoch=epoch, count=total, loss=loss_sum / total,
                       train_gain_mae=error_sum / total, train_seconds=time.monotonic() - begin)
            improved = False
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                # Retry interrupted epochs without overwriting a partial validation directory.
                folder = output / ('val-epoch-%03d' % epoch)
                attempt = 0
                while folder.exists():
                    attempt += 1
                    folder = output / ('val-epoch-%03d-attempt-%02d' % (epoch, attempt))
                result = validate_a(model.eval(), base, val, args.val_ann, folder, args.device,
                                    dict(epoch=epoch, label_fingerprint=data.labels.fingerprint))
                row['validation'] = result['metrics']
                ap = result['metrics']['coco/AP']
                improved = best_ap is None or ap > best_ap
                if improved:
                    best_ap = ap
            history.append(row)
            payload = dict(schema_version=CHECKPOINT_SCHEMA, input_schema=INPUT_SCHEMA,
                           trained=True, epoch=epoch, model_config=model.config,
                           model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                           base_checkpoint_sha256=EXPECTED_CHECKPOINT, settings=settings,
                           train_image_ids=train_images, best_val_ap=best_ap, history=history,
                           torch_rng_state=torch.get_rng_state(),
                           cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            save_checkpoint(output / 'last.pt', payload)
            if improved:
                save_checkpoint(output / 'best.pt', payload)
            atomic_json(output / 'history.json', history)
            print(json.dumps(row), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
