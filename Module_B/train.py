"""Train on train2017; select the epoch with highest full val2017 COCO AP."""
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib.metadata
import json
import random
import shutil
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from Module_B.config import (ALPHAS, FEATURE_SPEC, GAIN_SCALE, NETWORK,
                             DETECTION_CACHE, ROOT, EXPECTED_DETECTIONS)
from Module_B.common import sha, digest, atomic_json, source_hashes
from Module_B.data import prepare_data, normalization
from Module_B.model import get_model
from Module_B.validate import ValEvaluator, improves, PROTOCOL, TRAIN_FILES


def copy_exact(source, target):
    source, target = Path(source), Path(target)
    tmp = target.with_suffix(target.suffix+'.tmp')
    shutil.copyfile(source, tmp)
    if sha(source) != sha(tmp):
        raise ValueError('Checkpoint/report copy failed')
    os.replace(tmp, target)


def restore_aliases(output, state):
    """last.pt is the commit point; rebuild best/history after a partial write."""
    atomic_json(output/'history.json', state['history'])
    if state['best_epoch']:
        best = output/'epochs'/f"epoch_{state['best_epoch']:03d}.pt"
        copy_exact(best, output/'best.pt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT)
    for name in ('cache', 'labels', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--val-cache', type=Path, default=DETECTION_CACHE)
    p.add_argument('--ann', type=Path)
    p.add_argument('--checkpoint', type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--recover-lock', action='store_true',
        help='With --resume, recover an interrupted local writer after checking its PID is gone')
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.ann = args.ann or args.repo.parent/'data/coco/annotations/person_keypoints_val2017.json'
    args.checkpoint = args.checkpoint or args.repo/'checkpoints/cigpose-m_coco_256x192.pth'
    args.epochs = args.epochs if args.epochs is not None else (1 if args.smoke else 30)
    if args.epochs <= 0 or args.batch_size <= 0:
        p.error('Epochs and batch size must be positive')
    if args.recover_lock and not args.resume:
        p.error('--recover-lock requires --resume')
    for name in ('cache','labels','output','val_cache','ann','checkpoint'):
        setattr(args, name, getattr(args,name).resolve())
    for protected in (args.cache,args.labels,args.val_cache):
        if args.output == protected or protected in args.output.parents or args.output in protected.parents:
            raise ValueError('Use a separate new training output directory')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Output exists; use a new directory or identical --resume')
    if args.resume and not (args.output/'run.json').is_file():
        raise FileNotFoundError('Resume requires an existing aligned run')
    mmroot = args.repo/'mmpose'
    sys.path[:0] = [str(mmroot),str(mmroot/'analysis/fyp'),str(args.repo/'scripts')]
    os.chdir(mmroot)
    from shared_cache import ReferenceCache
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this aligned experiment')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache = ReferenceCache(args.cache)
    x,g,ann_ids,image_ids,dev,files,label_run,label_summary = prepare_data(
        cache,args.labels,args.seed,4096 if args.smoke else None,split_mode='all')
    if dev.any() or (args.full and len(ann_ids) != 149813):
        raise ValueError('Aligned full training must use all 149813 people and no internal holdout')
    mean,std = normalization(x,np.ones(len(x),dtype=bool))
    validator = ValEvaluator(args.repo,args.val_cache,args.ann,args.checkpoint,
                             set(map(int,image_ids)),args.device)
    split = dict(schema_version='b_train_val_ap_v1',protocol=PROTOCOL,
        dataset_split='train2017',scope='smoke' if args.smoke else 'full',
        train_annotation_ids=ann_ids.tolist(),train_image_ids=sorted(set(map(int,image_ids))),
        development_annotation_ids=[],development_image_ids=[],
        validation_image_ids=sorted(validator.images),validation_dataset='val2017',
        normalization_source='all selected train2017 rows',
        purpose='val2017 selects best epoch; this is not an independent test')
    model = get_model(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    run = dict(schema_version='b-gain-regressor-ap-v1',protocol=PROTOCOL,
        scope='smoke' if args.smoke else 'full',script_sha256=sha(__file__),
        code_sha256=source_hashes(*TRAIN_FILES),source_cache_run_sha256=sha(args.cache/'run.json'),
        labels_fingerprint=digest(label_run),labels_summary_sha256=sha(args.labels/'summary.json'),
        label_files_digest=digest(files),split_sha256=digest(split),feature_spec=FEATURE_SPEC,
        network=NETWORK,alphas=ALPHAS,gain_scale=GAIN_SCALE,
        loss='SmoothL1(beta=1) on all four gains multiplied by 1000; retain ties',
        optimizer=dict(type='AdamW',lr=1e-3,weight_decay=1e-4),
        model_selection='maximum unrounded full val2017 COCO AP; exact ties keep earlier epoch',
        validation=validator.provenance,epochs=args.epochs,batch_size=args.batch_size,seed=args.seed,
        device=args.device,device_name=torch.cuda.get_device_name(0),python=sys.version,
        versions={k:importlib.metadata.version(k) for k in
                  ('torch','numpy','mmpose','mmcv','mmengine','pycocotools')},
        CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        deterministic_algorithms=True,matmul_tf32=False,cudnn_tf32=True,
        cudnn_benchmark=False,cudnn_deterministic=True,
        train_people=len(ann_ids),internal_development_people=0,
        frozen_cigpose=True,ground_truth_in_inputs=False,
        val2017_used=True,val2017_role='epoch_selection_only',
        val2017_updates_parameters=False,independent_test=False,
        limitations=['Smoke trains on about 4096 people but validates on the entire val2017 set.',
            'Full mode trains all 149813 approved train2017 people.',
            'Reported best AP is a selected validation result, not independent test performance.',
            'B selects four positive strengths, with no reject action.'])
    if args.resume:
        if json.loads((args.output/'run.json').read_text()) != run:
            raise ValueError('Resume code/protocol/source/settings differ')
        if json.loads((args.output/'split.json').read_text()) != split:
            raise ValueError('Resume split changed')
    else:
        args.output.mkdir(parents=True)
        atomic_json(args.output/'run.json',run);atomic_json(args.output/'split.json',split)
    epochs_dir = args.output/'epochs';epochs_dir.mkdir(exist_ok=True)
    lock = args.output/'.writer.lock'
    if args.recover_lock and lock.exists():
        owner = json.loads(lock.read_text())
        import socket
        if owner.get('host') != socket.gethostname():
            raise ValueError('Cannot recover a lock from another host')
        try:os.kill(int(owner['pid']),0)
        except ProcessLookupError:lock.unlink()
        else:raise ValueError('Recorded training PID still exists; do not start another writer')
    import socket
    with lock.open('x') as handle:
        json.dump(dict(pid=os.getpid(),host=socket.gethostname()),handle)
    try:
        x -= mean;x /= std
        if not np.isfinite(x).all():raise ValueError('Non-finite normalized training input')
        train_x = torch.from_numpy(x)
        train_y = torch.from_numpy((g*GAIN_SCALE).astype(np.float32))
        del g
        generator = torch.Generator().manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x,train_y),
            batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=False,generator=generator)
        history=[];best_ap=-1.;best_epoch=0;start=0
        def state_dict(epoch):
            return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                best_AP=best_ap,best_epoch=best_epoch,history=history,
                normalization_mean=torch.from_numpy(mean),normalization_std=torch.from_numpy(std),
                feature_spec=FEATURE_SPEC,alphas=ALPHAS,gain_scale=GAIN_SCALE,
                protocol=PROTOCOL,run_fingerprint=digest(run),
                generator_state=generator.get_state(),rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all())
        def save_epoch(epoch):
            path=epochs_dir/f'epoch_{epoch:03d}.pt';tmp=path.with_suffix('.pt.tmp')
            torch.save(state_dict(epoch),tmp);os.replace(tmp,path)
            # last.pt is published after the immutable epoch snapshot is complete.
            copy_exact(path,args.output/'last.pt')
            restore_aliases(args.output,state_dict(epoch))
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=True)
            if state['run_fingerprint']!=digest(run) or state.get('protocol')!=PROTOCOL:
                raise ValueError('Resume checkpoint provenance mismatch')
            np.testing.assert_array_equal(state['normalization_mean'].numpy(),mean)
            np.testing.assert_array_equal(state['normalization_std'].numpy(),std)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            generator.set_state(state['generator_state']);torch.set_rng_state(state['rng_state'])
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])
            start=state['epoch'];best_ap=state['best_AP'];best_epoch=state['best_epoch'];history=state['history']
            if len(history)!=start:raise ValueError('Checkpoint history/epoch mismatch')
            restore_aliases(args.output,state)
            print(f'Resuming after epoch {start}; best epoch={best_epoch}, AP={100*best_ap:.6f}',flush=True)
        else:save_epoch(0)
        print(f'Training rows per epoch: {len(train_x)}; internal development rows: 0',flush=True)
        print('Selection: full val2017 COCO AP every epoch; exact ties keep earlier epoch',flush=True)
        for epoch in range(start+1,args.epochs+1):
            if source_hashes(*TRAIN_FILES)!=run['code_sha256']:
                raise ValueError('Training/validation code changed during run')
            model.train();total=0.;rows=0
            for bx,by in loader:
                bx=bx.to(args.device);by=by.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.smooth_l1_loss(model(bx),by)
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                total+=float(loss.detach())*len(bx);rows+=len(bx)
            if rows!=len(ann_ids):raise ValueError('An epoch did not train every selected person')
            print(f'Epoch {epoch}/{args.epochs}: trained {rows} people; starting full val2017 AP',flush=True)
            report,_=validator.evaluate(model,mean,std,GAIN_SCALE,args.output,epoch)
            ap=report['validation_AP']
            history.append(dict(epoch=epoch,train_people=rows,train_scaled_huber_loss=total/rows,
                                validation=report))
            if improves(ap,best_ap):best_ap=ap;best_epoch=epoch
            save_epoch(epoch)
            print(f'Epoch {epoch}/{args.epochs}: loss={total/rows:.6f}, val AP={100*ap:.6f}, '
                  f'best epoch={best_epoch}, best AP={100*best_ap:.6f}',flush=True)
        state=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'])
        checked,prediction_file=validator.evaluate(model,mean,std,GAIN_SCALE,args.output,best_epoch,verify=True)
        np.testing.assert_allclose(checked['validation_AP'],best_ap,rtol=0,atol=1e-12)
        original=history[best_epoch-1]['validation']['metrics']
        for k,v in checked['metrics'].items():np.testing.assert_allclose(v,original[k],rtol=0,atol=1e-12)
        copy_exact(prediction_file,args.output/'best_predictions.keypoints.json')
        atomic_json(args.output/'best_validation.json',checked)
        summary=dict(training_completed=True,scope=run['scope'],flow_check_only=bool(args.smoke),
            protocol=PROTOCOL,run_fingerprint=digest(run),instances=len(ann_ids),train_people=len(ann_ids),
            development_people=0,normalization_train_people=len(ann_ids),train_val_images_disjoint=True,
            gt_in_inputs=False,val2017_used=True,val2017_role='epoch_selection_only',
            val2017_updates_parameters=False,independent_test=False,
            best_epoch=best_epoch,best_validation_AP=best_ap,
            best_validation_metrics=checked['metrics'],best_checkpoint_sha256=sha(args.output/'best.pt'),
            best_predictions_sha256=sha(args.output/'best_predictions.keypoints.json'),
            validation_detections=EXPECTED_DETECTIONS,validation_images=5000,
            formal_coco_ap_evaluated=True,best_recheck_matches=True)
        atomic_json(args.output/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    finally:lock.unlink()


if __name__=='__main__':
    main()
