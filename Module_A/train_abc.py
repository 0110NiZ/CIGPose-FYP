"""Train C from actual frozen AB candidates; python Module_A/train_abc.py."""
import argparse
import contextlib
import fcntl
import json
import math
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import common
import config
from a_runtime import setup,load_frozen_base,require_cache_model,require_val_cache
from shared_cache import ReferenceCache,hash_file,atomic_json,digest_json
from detection_cache import ReferenceCache as DetectionCache
from evaluate import load_model
from model_ab import BRegressor,make_features,choose_alpha,ARCHITECTURE
from train_ab import a_choices
from model_abc import ModuleCAcceptanceClassifier
from evaluate_abc import c_inputs,fallback
from build_shared_cache import candidate_from_cache
from build_a_labels_legacy import batch_oks,get_gt
from predict_a import prediction_records
from eval_unified import UnifiedPoseEvaluator
from result import update_experiment,read_result


@torch.inference_mode()
def candidate_data(a,b,base,ar,device,annotations=None):
    joints=a_choices(a,ar,device)
    alpha=choose_alpha(b(torch.as_tensor(make_features(ar,joints.cpu().numpy()),device=device)))
    pc,sc=candidate_from_cache(base.head,ar,joints,alpha,device)
    out=dict(x=c_inputs(ar,pc,sc,joints,alpha,device).cpu().numpy(),pc=pc,sc=sc,
             joint_ids=joints.cpu().numpy(),alpha=alpha.cpu().numpy(),image_id=ar['image_id'])
    id_key='annotation_id' if annotations is not None else 'source_detection_index'
    out[id_key]=ar[id_key]
    if annotations is not None:
        gt,area=get_gt(ar,annotations)
        out['gain']=batch_oks(pc,gt,area)-batch_oks(ar['P_0_coords'],gt,area)
        out['y']=(out['gain']>0).astype(np.float32)
    return out


def prepare(cache,folder,identity,a,b,base,device,training):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'writer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=folder/'metadata.json'
        if meta.exists():
            if json.loads(meta.read_text())!=identity:raise ValueError('缓存来源改变，请指定新的 --cache-dir')
        else:atomic_json(meta,identity)
        annotations=None
        if training:annotations={int(r['id']):r for r in json.loads((cache.directory/'selected_annotations.json').read_text())['annotations']}
        paths=cache.shards();seen=set();chunks=[];key='annotation_id' if training else 'source_detection_index'
        for i,path in enumerate(paths,1):
            ar=cache.load_shard(path);dest=folder/path.name;receipt=dest.with_suffix('.json');source=hash_file(path)
            if dest.exists() and receipt.exists():
                if json.loads(receipt.read_text())!=dict(source_sha256=source,payload_sha256=hash_file(dest)):
                    raise ValueError('候选缓存校验失败: '+str(dest))
                with np.load(dest,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
            else:
                d=candidate_data(a,b,base,ar,device,annotations)
                tmp=dest.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**d)
                tmp.replace(dest);atomic_json(receipt,dict(source_sha256=source,payload_sha256=hash_file(dest)))
            np.testing.assert_array_equal(d[key],ar[key]);np.testing.assert_array_equal(d['image_id'],ar['image_id'])
            if d['x'].shape!=(len(ar[key]),137) or not all(np.isfinite(v).all() for v in d.values()):raise ValueError('Invalid candidate payload')
            if training:np.testing.assert_array_equal(d['y'],(d['gain']>0).astype(np.float32))
            elif 'gain' in d or 'y' in d:raise ValueError('Validation candidate cache must not contain GT labels')
            ids=set(map(int,ar[key]))
            if seen&ids:raise ValueError('Duplicate people')
            seen.update(ids)
            if training:chunks.append({k:d[k] for k in ('x','y','gain')})
            if i==1 or i%200==0 or i==len(paths):print('%s候选 [%d/%d] %d/%d'%('训练' if training else '验证',i,len(paths),len(seen),len(cache.expected)),flush=True)
        if seen!=set(cache.expected):raise ValueError('Incomplete people')
        atomic_json(folder/'complete.json',dict(count=len(seen),fingerprint=digest_json(identity)))
        return {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]} if training else None


def weights(gain,scale):
    w=np.minimum(np.abs(gain.astype(np.float32))/scale,1.).astype(np.float32)
    if not np.isfinite(w).all() or w.mean()<=0:raise ValueError('Invalid/zero training weights')
    return w/w.mean()


@torch.inference_mode()
def validate(c,cache,folder,tau,device,label,always_accept=False):
    c.eval();evaluator=UnifiedPoseEvaluator(config.VAL_ANN);seen=set();accepted=0;paths=cache.shards()
    for i,path in enumerate(paths,1):
        ar=cache.load_shard(path);dest=folder/path.name
        if json.loads(dest.with_suffix('.json').read_text())!=dict(source_sha256=hash_file(path),payload_sha256=hash_file(dest)):
            raise ValueError('Validation cache changed')
        with np.load(dest,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['source_detection_index'],ar['source_detection_index'])
            np.testing.assert_array_equal(z['image_id'],ar['image_id'])
            prob=np.ones(len(z['x'])) if always_accept else c.predict_proba(torch.as_tensor(z['x'],device=device)).cpu().numpy()
            coords,scores,mask=fallback(ar,z['pc'],z['sc'],prob,tau)
        ids=set(map(int,ar['source_detection_index']))
        if seen&ids:raise ValueError('Duplicate detection')
        seen.update(ids);accepted+=int(mask.sum())
        evaluator.add_records(list(prediction_records(cache,ar,coords,scores)))
        if i==1 or i%400==0 or i==len(paths):print('%s 验证 [%d/%d] %d/104125，接受率 %.2f%%'%(label,i,len(paths),len(seen),100*accepted/len(seen)),flush=True)
    if seen!=set(cache.expected):raise ValueError('Incomplete validation')
    print(label+' 计算完整AP…',flush=True)
    with tempfile.TemporaryFile(mode='w+') as log,contextlib.redirect_stdout(log):metrics,audit=evaluator.evaluate()
    assert audit['num_input_instances']==104125 and audit['evaluation_image_count']==5000
    return metrics,audit,accepted/len(seen)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ab-checkpoint',type=Path,default=config.WORK/'runs/model_ab_seed42/best.pt')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--gain-scale',type=float,default=.001)
    p.add_argument('--tau',type=float,default=.51);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default=config.DEVICE);p.add_argument('--output',type=Path)
    p.add_argument('--cache-dir',type=Path,default=config.WORK/'labels/abc_ab_candidates_v1')
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.seed<0 or not all(math.isfinite(x) for x in (args.lr,args.gain_scale,args.tau)) or min(args.lr,args.gain_scale)<=0 or not 0<=args.tau<1:p.error('Invalid settings')
    output=(args.output or config.WORK/'runs'/('model_abc_seed%d'%args.seed)).resolve();folder=args.cache_dir.resolve()
    protected=[config.TRAIN_CACHE.resolve(),config.VAL_CACHE.resolve(),config.LABELS.resolve(),args.ab_checkpoint.parent.resolve()]
    for dest in (output,folder):
        if any(dest==q or dest in q.parents or q in dest.parents for q in protected):raise ValueError('Use separate new output/cache directories')
    if output==folder or output in folder.parents or folder in output.parents:raise ValueError('Separate output and cache')
    if output.exists() and not args.check_only:raise FileExistsError('输出目录已存在；使用 --output 新目录，不覆盖已有结果。')
    setup(args.seed);ab=torch.load(args.ab_checkpoint,map_location='cpu');a_path=Path(ab['settings']['a_checkpoint'])
    if ab.get('architecture')!=ARCHITECTURE or not ab.get('trained'):raise ValueError('Expected trained AB')
    if hash_file(a_path)!=ab['settings']['identity']['a_sha256']:raise ValueError('A checkpoint mismatch')
    a_saved=torch.load(a_path,map_location='cpu');a=load_model(a_saved,args.device).requires_grad_(False).eval()
    b=BRegressor().to(args.device);b.load_state_dict(ab['model_state']);b.requires_grad_(False).eval()
    base=load_frozen_base(config.BASE_MODEL,args.device)
    if hash_file(config.BASE_MODEL)!=ab['settings']['identity']['base_sha256']:raise ValueError('Base mismatch')
    train=ReferenceCache(config.TRAIN_CACHE);require_cache_model(train)
    val=DetectionCache(config.VAL_CACHE);require_val_cache(val)
    if val.fingerprint!=ab['settings']['val_cache_fingerprint']:raise ValueError('Use the same validation cache as AB')
    val_images={int(r['id']) for r in json.loads(config.VAL_ANN.read_text())['images']}
    if len(train.expected)!=149813 or train.run['protocol']['dataset_split']!='train2017' or len(val_images)!=5000 or set(train.expected.values())&val_images or set(a_saved.get('train_image_ids',[]))&val_images:raise ValueError('Dataset split/coverage mismatch')
    if hash_file(config.VAL_ANN)!=val.run['protocol']['source_annotation_sha256']:raise ValueError('Validation annotation mismatch')
    identity=dict(version='abc_ab_candidates_v1',a_sha256=hash_file(a_path),ab_sha256=hash_file(args.ab_checkpoint),base_sha256=hash_file(config.BASE_MODEL),
        p0_reference='original cache P_0_coords',source_sha256={str(q):hash_file(q) for q in [Path(__file__),Path(__file__).with_name('model_abc.py'),Path(__file__).with_name('evaluate_abc.py'),Path(__file__).with_name('model_ab.py'),Path(__file__).with_name('model_a3.py'),Path(__file__).with_name('train_ab.py'),Path(common.PUBLIC)/'a_contract.py',Path(common.PUBLIC)/'build_shared_cache.py',Path(common.PUBLIC)/'precision_policy.py',Path(common.PUBLIC)/'build_a_labels_legacy.py']})
    print('冻结A3和最佳B，只从头训练C；固定阈值 %.2f；默认完整验证每轮一次。'%args.tau,flush=True)
    if args.check_only:
        ar=train.load_shard(train.shards()[0]);ann={int(r['id']):r for r in json.loads((train.directory/'selected_annotations.json').read_text())['annotations']}
        before=[{k:v.clone() for k,v in m.state_dict().items()} for m in (a,b,base)]
        d=candidate_data(a,b,base,ar,args.device,ann);c=ModuleCAcceptanceClassifier(137,128,.1).to(args.device)
        w=weights(d['gain'],args.gain_scale);opt=torch.optim.AdamW(c.parameters(),lr=args.lr)
        assert abs(float(w.mean())-1)<1e-6
        np.testing.assert_array_equal(weights(np.array([0.,-.001,.002]),.001),np.array([0.,1.5,1.5],np.float32))
        c_before=[v.clone() for v in c.parameters()]
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(c(torch.as_tensor(d['x'],device=args.device)),torch.as_tensor(d['y'],device=args.device),reduction='none')*torch.as_tensor(w,device=args.device)).mean()
        loss.backward();opt.step();assert torch.isfinite(loss)
        assert any(not torch.equal(v,old) for v,old in zip(c.parameters(),c_before))
        for model,old in zip((a,b,base),before):
            assert all(v.grad is None for v in model.parameters()) and all(torch.equal(v,old[k]) for k,v in model.state_dict().items())
        va=val.load_shard(val.shards()[0]);vd=candidate_data(a,b,base,va,args.device)
        assert 'gain' not in vd and 'y' not in vd
        np.testing.assert_array_equal(fallback(va,vd['pc'],vd['sc'],np.zeros(len(vd['x'])),args.tau)[0],va['P_0_coords'])
        with tempfile.TemporaryDirectory() as temp:
            from types import SimpleNamespace
            path=train.shards()[0];small=SimpleNamespace(directory=train.directory,expected={int(i):int(j) for i,j in zip(ar['annotation_id'],ar['image_id'])},shards=lambda:[path],load_shard=train.load_shard)
            one=prepare(small,Path(temp),identity,a,b,base,args.device,True)
            two=prepare(small,Path(temp),identity,a,b,base,args.device,True)
            for k in one:np.testing.assert_array_equal(one[k],two[k])
            receipt=(Path(temp)/path.name).with_suffix('.json');bad=json.loads(receipt.read_text());bad['payload_sha256']='invalid';atomic_json(receipt,bad)
            try:prepare(small,Path(temp),identity,a,b,base,args.device,True)
            except ValueError as e:
                if '校验失败' not in str(e):raise
            else:raise AssertionError('Corrupt cache accepted')
        print('PASS：真实32人训练/验证候选，标签与权重，缓存复用，C更新而A/B/CIGPose不变；未运行全量训练/AP。',flush=True);return
    print('检查全量缓存…',flush=True);train.verify(True);val.verify(True)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'writer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train_id=dict(identity,cache_fingerprint=train.fingerprint,annotations_sha256=hash_file(train.directory/'selected_annotations.json'))
        val_id=dict(identity,cache_fingerprint=val.fingerprint)
        data=prepare(train,folder/'train',train_id,a,b,base,args.device,True)
        prepare(val,folder/'val',val_id,a,b,base,args.device,False)
        setup(args.seed);c=ModuleCAcceptanceClassifier(137,128,.1).to(args.device)
        reference,_,_=validate(c,val,folder/'val',args.tau,args.device,'AB一致性复核',True)
        for k,v in ab['validation'].items():
            if abs(reference[k]-v)>1e-12:raise ValueError('AB parity failed: '+k)
        w=weights(data['gain'],args.gain_scale)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(data['x']),torch.from_numpy(data['y']),torch.from_numpy(w)),batch_size=args.batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
        optimizer=torch.optim.AdamW(c.parameters(),lr=args.lr,weight_decay=.0001)
        settings=dict(identity=identity,train_identity=train_id,val_identity=val_id,ab_checkpoint=str(args.ab_checkpoint),a_checkpoint=str(a_path),epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,seed=args.seed,gain_scale=args.gain_scale,tau=args.tau,threshold_policy='fixed before training; no threshold sweep',train_people=len(w),train_positive=int(data['y'].sum()),train_zero_weight=int((w==0).sum()),ab_reference=reference)
        atomic_json(output/'settings.json',settings)
        history=[];best=-float('inf');best_epoch=None
        for epoch in range(1,args.epochs+1):
            c.train();total=0.;count=0;label='Epoch %d/%d'%(epoch,args.epochs)
            for i,(x,y,wt) in enumerate(loader,1):
                x=x.to(args.device);y=y.to(args.device);wt=wt.to(args.device);optimizer.zero_grad(set_to_none=True)
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(c(x),y,reduction='none')*wt).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(c.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                count+=len(x);total+=float(loss.detach())*len(x)
                if i==1 or i%100==0 or i==len(loader):print('%s 训练 [%d/%d] %d/149813 Loss=%.6f'%(label,i,len(loader),count,total/count),flush=True)
            assert count==149813
            metrics,audit,rate=validate(c,val,folder/'val',args.tau,args.device,label)
            ap=metrics['coco/AP'];improved=ap>best
            if improved:
                best=ap;best_epoch=epoch
                payload=dict(epoch=epoch,model_state={k:v.detach().cpu().clone() for k,v in c.state_dict().items()},input_dim=137,hidden_dim=128,dropout=.1,training='ab_gain_weighted_bce',gain_scale=args.gain_scale,tau=args.tau,settings=settings,validation=metrics)
                tmp=output/'best.pt.tmp';torch.save(payload,tmp);tmp.replace(output/'best.pt')
            history.append(dict(epoch=epoch,count=count,loss=total/count,validation=metrics,audit=audit,accept_rate=rate))
            update_experiment(output.name,dict(title='固定A+B，重新训练C',description='C从头训练；阈值固定%.2f；每轮完整验证，按AP保留最佳。'%args.tau,settings=settings,history=history,best_epoch=best_epoch,checkpoint=str(output/'best.pt')))
            print('%s 完成 AP=%.6f 接受率=%.2f%% 最佳=第%d轮 %.6f%s'%(label,100*ap,100*rate,best_epoch,100*best,' [新最佳]' if improved else ''),flush=True)
        saved=torch.load(output/'best.pt',map_location='cpu');c.load_state_dict(saved['model_state'])
        checked,audit,rate=validate(c,val,folder/'val',args.tau,args.device,'最佳模型复核')
        for k,v in saved['validation'].items():
            if abs(checked[k]-v)>1e-12:raise ValueError('Best C recheck mismatch')
        assert hash_file(a_path)==identity['a_sha256'] and hash_file(args.ab_checkpoint)==identity['ab_sha256']
        update_experiment(output.name,dict(title='固定A+B，重新训练C',description='完整训练结束，最佳权重复核通过；阈值固定%.2f。'%args.tau,settings=settings,history=history,best_epoch=best_epoch,checkpoint=str(output/'best.pt'),completed=True,best_recheck=checked))
        print('\n最终结果：轮次 / Loss / AP / 接受率 / 标记',flush=True)
        for r in history:print('%d / %.6f / %.6f / %.2f%% / %s'%(r['epoch'],r['loss'],100*r['validation']['coco/AP'],100*r['accept_rate'],'最终最佳' if r['epoch']==best_epoch else ''),flush=True)
        print('AB: %.6f；重新训练C后的ABC: %.6f；差值: %+.6f AP点'%(100*reference['coco/AP'],100*best,100*(best-reference['coco/AP'])),flush=True)
        for k,v in checked.items():print('%s = %.6f'%(k,100*v),flush=True)
        print('模型：%s\n统一结果：%s'%(output/'best.pt',config.WORK/'code/experiments/module_a/results.html'),flush=True)


if __name__=='__main__':main()
