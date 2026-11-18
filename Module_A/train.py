"""Shared training loop: full validation every epoch, save best.pt only.

Use train_a3.py, train_local_small.py, etc. to select an experiment.
Each train_<name>.py supplies only its model, optimizer defaults and optional loss.
"""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import importlib
import numpy as np
import torch
from torch.nn import functional as F
import common
import config
from a_contract import CHECKPOINT_SCHEMA,INPUT_SCHEMA,INPUT_KEYS,model_inputs
from a_runtime import setup,require_cache_model,require_val_cache,load_frozen_base,EXPECTED_CHECKPOINT
from data import ATrainingData,fit_input_statistics
from detection_cache import ReferenceCache
from shared_cache import hash_file
from model import to_device, MLP  # MLP remains an interface-test compatibility export.
from evaluate import evaluate_model
from result import update_experiment


def smooth_l1(pred,target):
    return F.smooth_l1_loss(pred*100,target*100)


# Public interface tests still import these historical names.
gain_loss=smooth_l1
DetectionCache=ReferenceCache


def load_full_inputs(data,device):
    n=len(data.labels.ids)
    raw={k:np.empty((n,)+shape,np.float32) for k,shape in {
        'features':(17,256),'uncertainty':(17,),'pose_coords':(17,2),'pose_scores':(17,)}.items()}
    labels=np.empty((n,17),np.float32);seen=[];offset=0
    for path in data.cache.shards():
        arrays=data.cache.load_shard(path);inputs=model_inputs(arrays);size=len(arrays['annotation_id'])
        for key in INPUT_KEYS:raw[key][offset:offset+size]=inputs[key]
        labels[offset:offset+size]=data.labels.lookup(arrays['annotation_id'],arrays['image_id'])
        seen.extend(arrays['annotation_id'].tolist());offset+=size
    assert offset==n and len(set(seen))==n and set(seen)==set(data.labels.ids)
    stats=np.stack((raw['uncertainty'].astype(np.float64),raw['pose_scores'].astype(np.float64)))
    mean=stats.mean((1,2)).astype(np.float32);std=np.maximum(stats.std((1,2)),1e-6).astype(np.float32)
    inputs={key:torch.as_tensor(raw.pop(key),device=device) for key in INPUT_KEYS}
    return inputs,torch.as_tensor(labels,device=device),labels,mean,std


def run(model_module,title,default_lr=.001,loss_fn=smooth_l1,order='global'):
    parser=argparse.ArgumentParser(description=title)
    parser.add_argument('--run-training',action='store_true')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--lr',type=float,default=default_lr)
    parser.add_argument('--weight-decay',type=float,default=.0001)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default=config.DEVICE)
    args=parser.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.lr<=0 or args.weight_decay<0 or args.seed<0:
        parser.error('Invalid training settings')
    if not np.isfinite([args.lr,args.weight_decay]).all():parser.error('Nonfinite optimizer settings')
    output=(args.output or config.WORK/'runs'/('%s_seed%d'%(model_module,args.seed))).resolve()
    if not args.run_training:
        print(title);print('Output:',output);print('Full validation every epoch; best.pt only. Add --run-training to start.')
        return
    if output.exists():raise FileExistsError('Use a new output directory: '+str(output))
    precision=setup(args.seed)
    data=ATrainingData(config.TRAIN_CACHE,config.LABELS);require_cache_model(data.cache)
    val=ReferenceCache(config.VAL_CACHE);require_val_cache(val);val.verify(True)
    train_images=sorted(set(data.cache.expected.values()))
    val_images={int(i['id']) for i in json.loads(config.VAL_ANN.read_text())['images']}
    assert len(val_images)==5000 and not set(train_images).intersection(val_images)
    assert hash_file(config.BASE_MODEL)==EXPECTED_CHECKPOINT
    assert hash_file(config.VAL_ANN)==val.run['protocol']['source_annotation_sha256']
    n=len(data.labels.ids)
    # Initialize before loading the base so its initialization cannot change A's seed.
    model=importlib.import_module(model_module).MLP().to(args.device)
    if order=='global':
        x,y,target,mean,std=load_full_inputs(data,args.device)
    elif order=='buffered':
        mean,std=fit_input_statistics(data.cache)
    else:raise ValueError('Unknown data order')
    model.set_input_statistics(mean,std)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
    output.mkdir(parents=True)
    from mmengine.logging import MMLogger
    MMLogger.get_current_instance()
    with contextlib.redirect_stdout(__import__('sys').stderr):base=load_frozen_base(config.BASE_MODEL,args.device)
    settings=dict(model_module=model_module,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,
        weight_decay=args.weight_decay,seed=args.seed,order=order,loss=loss_fn.__name__,
        train_people=n,train_cache_fingerprint=data.cache.fingerprint,label_fingerprint=data.labels.fingerprint,
        val_cache_fingerprint=val.fingerprint,precision=precision,validation='every epoch, full5000/104125',
        code_hashes={p.name:hash_file(p) for p in Path(__file__).resolve().parent.glob('*.py')})
    history=[];best=-float('inf');best_epoch=None
    for epoch in range(1,args.epochs+1):
        model.train();total=0;loss_sum=0.
        if order=='global':
            permutation=np.random.default_rng(np.random.SeedSequence([args.seed,epoch])).permutation(n)
            def batches():
                for begin in range(0,n,args.batch_size):
                    idx=torch.as_tensor(permutation[begin:begin+args.batch_size],device=args.device)
                    yield {k:v[idx] for k,v in x.items()},y[idx]
        else:
            def batches():
                for inputs,labels,ids in data.batches(args.batch_size,args.seed,epoch,32):
                    yield to_device(inputs,args.device),torch.as_tensor(labels,device=args.device)
        for inputs,labels in batches():
            prediction=model(**inputs);loss=loss_fn(prediction,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);optimizer.step()
            total+=len(labels);loss_sum+=float(loss.detach())*len(labels)
        assert total==n
        row=dict(epoch=epoch,count=total,loss=loss_sum/total)
        if order=='global':
            gain_sum=0.
            with torch.inference_mode():
                model.eval()
                for begin in range(0,n,args.batch_size):
                    pred=model(**{k:v[begin:begin+args.batch_size] for k,v in x.items()})
                    choices=pred.argmax(1)
                    gain_sum+=float(y[begin:begin+args.batch_size].gather(1,choices[:,None]).double().sum())
            row['train_mean_gain']=gain_sum/n
        validation=evaluate_model(model,base,val,args.device,dict(experiment=model_module,epoch=epoch))
        row['validation']=validation['metrics'];row['audit']=validation['audit'];history.append(row)
        ap=row['validation']['coco/AP']
        if ap>best:
            best=ap;best_epoch=epoch
            payload=dict(schema_version=CHECKPOINT_SCHEMA,input_schema=INPUT_SCHEMA,trained=True,
                model_module=model_module,model_config=getattr(model,'config',{}),
                model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                epoch=epoch,best_epoch=epoch,validation=row['validation'],settings=settings,
                train_image_ids=train_images,base_checkpoint_sha256=EXPECTED_CHECKPOINT)
            temporary=output/'best.pt.tmp';torch.save(payload,temporary);temporary.replace(output/'best.pt')
        record=dict(title=title,description='每轮完整验证；同分取最早轮；只保留最佳模型。',
            settings=settings,history=history,best_epoch=best_epoch,checkpoint=str(output/'best.pt'))
        update_experiment(output.name,record)
        print('%s epoch%d AP=%.6f best=epoch%d %.6f'%(title,epoch,ap*100,best_epoch,best*100),flush=True)
        gc.collect()


def main():
    """Legacy data-only preflight. Training always uses a named train_* entry."""
    import sys
    if len(sys.argv)==1:
        print('Use a named experiment: python train_a3.py --run-training --output NEW_DIRECTORY')
        return
    p=argparse.ArgumentParser(description='Legacy data-only interface check; no training')
    p.add_argument('--train-cache',type=Path,default=config.TRAIN_CACHE)
    p.add_argument('--labels',type=Path,default=config.LABELS)
    p.add_argument('--val-cache',type=Path,default=config.VAL_CACHE)
    p.add_argument('--val-ann',type=Path,default=config.VAL_ANN)
    p.add_argument('--base-checkpoint',type=Path,default=config.BASE_MODEL)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    data=ATrainingData(args.train_cache,args.labels);require_cache_model(data.cache)
    val=DetectionCache(args.val_cache);require_val_cache(val)
    ids={int(i['id']) for i in json.loads(args.val_ann.read_text())['images']}
    assert len(ids)==5000 and not set(data.cache.expected.values()).intersection(ids)
    assert hash_file(args.base_checkpoint)==EXPECTED_CHECKPOINT
    print('Data-only checks passed; no model, optimizer or output was created.')


if __name__=='__main__':main()
