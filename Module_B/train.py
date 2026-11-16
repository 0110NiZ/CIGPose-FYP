"""Train B on cached CIGPose features."""
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from Module_B.config import ALPHAS, FEATURE_SPEC, GAIN_SCALE, NETWORK
from Module_B.common import sha, digest, atomic_json, source_hashes
from Module_B.data import prepare_data, normalization, gain_metrics
from Module_B.model import get_model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache', 'labels', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (3 if args.smoke else 30)
    if args.epochs <= 0 or args.batch_size <= 0:
        p.error('Epochs and batch size must be positive')
    for name in ('repo', 'cache', 'labels', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    for protected in (args.cache, args.labels):
        if args.output == protected or protected in args.output.parents or args.output in protected.parents:
            raise ValueError('Training output must be separate from labels and source cache')
    sys.path.insert(0, str(args.repo/'mmpose/analysis/fyp'))
    from shared_cache import ReferenceCache
    cache = ReferenceCache(args.cache)
    x, g, ann_ids, image_ids, dev, files, labels_run, labels_summary = prepare_data(
        cache, args.labels, args.seed, 4096 if args.smoke else None)
    mean, std = normalization(x, ~dev)
    split = dict(schema_version='b_image_development_v1', dataset_split='train2017',
        purpose='B internal development only; official val2017 is untouched', seed=args.seed,
        rule='sha256(b-train-v1:seed:split:image_id) first 64 bits < floor(2**64/10)',
        development_fraction_target=.1, smoke=bool(args.smoke),
        train_annotation_ids=ann_ids[~dev].tolist(), development_annotation_ids=ann_ids[dev].tolist(),
        train_image_ids=sorted(set(map(int,image_ids[~dev]))),
        development_image_ids=sorted(set(map(int,image_ids[dev]))))
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    nn = torch.nn
    model = get_model(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scale = GAIN_SCALE
    run = dict(schema_version='b-gain-regressor-v1', scope='smoke' if args.smoke else 'full',
        script_sha256=sha(__file__), code_sha256=source_hashes('train.py','common.py','config.py','data.py','model.py'), source_cache_run_sha256=sha(args.cache/'run.json'),
        labels_fingerprint=digest(labels_run), labels_summary_sha256=sha(args.labels/'summary.json'),
        label_files_digest=digest(files), split_sha256=digest(split), feature_spec=FEATURE_SPEC,
        network=NETWORK, alphas=ALPHAS,
        loss='SmoothL1(beta=1) on each of four gains multiplied by 1000; retain ties',
        gain_scale=scale, optimizer=dict(type='AdamW',lr=1e-3,weight_decay=1e-4),
        model_selection='maximum mean selected gain on image-disjoint train2017 development',
        epochs=args.epochs,batch_size=args.batch_size,seed=args.seed,
        device=args.device,torch_version=torch.__version__,numpy_version=np.__version__,
        python=sys.version, CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        frozen_cigpose=True, ground_truth_in_inputs=False, val2017_used=False,
        limitations=['Smoke results only validate training flow.',
            'Development OKS gains are not formal COCO AP/AR.',
            'B selects only positive intervention strengths; no reject action.',
            'Official COCO evaluation requires matching current-runtime P0/Top-1 controls.'])
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Output exists; use a new directory or identical --resume run')
        if json.loads((args.output/'run.json').read_text()) != run:
            raise ValueError('Resume inputs/code/settings differ')
    else:
        if args.resume:
            raise FileNotFoundError('Cannot resume nonexistent output')
        args.output.mkdir(parents=True)
        atomic_json(args.output/'run.json',run)
        atomic_json(args.output/'split.json',split)
    lock = args.output/'.writer.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        # Statistics fitted to training rows only; B sees no target-dependent inputs.
        x -= mean
        x /= std
        if not np.isfinite(x).all():
            raise ValueError('Non-finite standardized feature')
        train_x = torch.from_numpy(x[~dev].copy())
        train_y = torch.from_numpy((g[~dev]*scale).astype(np.float32))
        dev_x = torch.from_numpy(x[dev].copy())
        dev_gain = g[dev]
        fixed = int(g[~dev].mean(axis=0).argmax())
        del x
        generator = torch.Generator().manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x,train_y),
            batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
        def evaluate():
            model.eval()
            predictions=[]
            with torch.inference_mode():
                for start in range(0,len(dev_x),1024):
                    predictions.append(model(dev_x[start:start+1024].to(args.device)).cpu().numpy()/scale)
            return gain_metrics(np.concatenate(predictions), dev_gain, fixed)
        def checkpoint(epoch,best_gain,best_epoch):
            return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                best_gain=best_gain,best_epoch=best_epoch,normalization_mean=torch.from_numpy(mean),
                normalization_std=torch.from_numpy(std),feature_spec=FEATURE_SPEC,alphas=ALPHAS,
                gain_scale=scale,run_fingerprint=digest(run),generator_state=generator.get_state(),
                rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        def save_checkpoint(name,payload):
            tmp=args.output/(name+'.tmp')
            torch.save(payload,tmp)
            os.replace(tmp,args.output/name)
        start,best_gain,best_epoch=0,-float('inf'),0
        history=[]
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=True)
            if state['run_fingerprint']!=digest(run):
                raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            generator.set_state(state['generator_state']);torch.set_rng_state(state['rng_state'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['cuda_rng_state'])
            start,best_gain,best_epoch=state['epoch'],state['best_gain'],state['best_epoch']
            if (args.output/'history.json').exists():
                history=json.loads((args.output/'history.json').read_text())[:start]
        for epoch in range(start+1,args.epochs+1):
            model.train();total=0.;rows=0
            for bx,by in loader:
                bx,by=bx.to(args.device),by.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.smooth_l1_loss(model(bx),by)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                total+=float(loss.detach())*len(bx);rows+=len(bx)
            metrics=evaluate()
            record=dict(epoch=epoch,train_scaled_huber_loss=total/rows,development=metrics)
            history.append(record)
            if metrics['selected_mean_gain']>best_gain:
                best_gain=metrics['selected_mean_gain'];best_epoch=epoch
                save_checkpoint('best.pt',checkpoint(epoch,best_gain,best_epoch))
            atomic_json(args.output/'history.json',history)
            save_checkpoint('last.pt',checkpoint(epoch,best_gain,best_epoch))
            print(f"Epoch {epoch}/{args.epochs}: loss={total/rows:.6f}, "
                  f"dev gain={metrics['selected_mean_gain']:+.8f}, "
                  f"vs train-fixed={metrics['mean_gain_vs_fixed_from_train']:+.8f}",flush=True)
        state=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'])
        metrics=evaluate()
        result=dict(training_completed=True,scope=run['scope'],flow_check_only=bool(args.smoke),
            run_fingerprint=digest(run),instances=len(ann_ids),train_people=int((~dev).sum()),
            development_people=int(dev.sum()),images_disjoint=True,gt_in_inputs=False,
            val2017_used=False,best_epoch=best_epoch,best_checkpoint_sha256=sha(args.output/'best.pt'),
            development=metrics,beats_train_selected_fixed=metrics['mean_gain_vs_fixed_from_train']>0,
            formal_coco_ap_evaluated=False)
        atomic_json(args.output/'summary.json',result)
        print(json.dumps(result,indent=2),flush=True)
    finally:
        lock.unlink()



if __name__ == '__main__':
    main()
