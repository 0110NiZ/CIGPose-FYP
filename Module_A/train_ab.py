"""python Module_A/train_ab.py: frozen A3, train only B; full AP every epoch."""
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
from model_ab import BRegressor, make_features, choose_alpha, ALPHAS, GAIN_SCALE, ARCHITECTURE
from model_a3 import to_device, select_joint
from a_contract import model_inputs
from a_runtime import setup, load_frozen_base, require_cache_model, require_val_cache, EXPECTED_CHECKPOINT
from shared_cache import ReferenceCache, hash_file, atomic_json, digest_json
from detection_cache import ReferenceCache as DetectionCache
from build_shared_cache import candidate_from_cache
from build_a_labels_legacy import batch_oks, get_gt
from evaluate import load_model
from result import read_result, update_experiment

A_KEY = 'module_a_a3_full_joint_context_seed42_bs1024_eval1'


@torch.inference_mode()
def a_choices(a, arrays, device):
    return select_joint(a(**to_device(model_inputs(arrays), device)))


@torch.inference_mode()
def label_batch(a, base, arrays, annotations, device):
    joints = a_choices(a, arrays, device)
    gt, area = get_gt(arrays, annotations)
    # Match A3: all four labels use the SAME original cached P0 reference.
    before = batch_oks(arrays['P_0_coords'], gt, area)
    gains = []
    for alpha in ALPHAS:
        coords, _ = candidate_from_cache(base.head, arrays, joints,
                                        torch.full((len(joints),), alpha, device=device), device)
        gains.append(batch_oks(coords, gt, area) - before)
    x = make_features(arrays, joints.cpu().numpy())
    return x, np.stack(gains, 1).astype(np.float32), joints.cpu().numpy()


def prepare_labels(a, base, cache, directory, identity, device):
    """Atomic verified shards: interrupted label generation is reusable, not training resume."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = directory/'metadata.json'
        if meta.exists():
            if json.loads(meta.read_text()) != identity:
                raise ValueError('Label provenance changed; choose another --labels directory')
        else:
            atomic_json(meta, identity)
        annotations = {int(r['id']): r for r in json.loads(
            (cache.directory/'selected_annotations.json').read_text())['annotations']}
        xs, ys, seen = [], [], set()
        paths = cache.shards()
        for i, path in enumerate(paths, 1):
            arrays = cache.load_shard(path)
            destination = directory/path.name
            receipt = destination.with_suffix('.json')
            source_sha = hash_file(path)
            if destination.exists() and receipt.exists():
                r = json.loads(receipt.read_text())
                if r != dict(source_sha256=source_sha, payload_sha256=hash_file(destination)):
                    raise ValueError('Label/source checksum mismatch: '+str(path))
                with np.load(destination, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['annotation_id'], arrays['annotation_id'])
                    np.testing.assert_array_equal(z['image_id'], arrays['image_id'])
                    np.testing.assert_array_equal(z['joint_ids'], a_choices(a, arrays, device).cpu().numpy())
                    x, y = z['x'].copy(), z['gain'].copy()
            else:
                x, y, joints = label_batch(a, base, arrays, annotations, device)
                tmp = destination.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    np.savez_compressed(f, x=x, gain=y, joint_ids=joints,
                                        annotation_id=arrays['annotation_id'], image_id=arrays['image_id'])
                tmp.replace(destination)
                atomic_json(receipt, dict(source_sha256=source_sha, payload_sha256=hash_file(destination)))
            n = len(arrays['annotation_id'])
            if x.shape != (n,563) or y.shape != (n,4) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Invalid AB training payload')
            ids = set(map(int, arrays['annotation_id']))
            if seen & ids: raise ValueError('Duplicate training IDs')
            seen.update(ids);xs.append(x);ys.append(y)
            if i == 1 or i % 100 == 0 or i == len(paths):
                print('标签 [%d/%d] 人体 %d/%d（A固定，每人四种强度）' %
                      (i,len(paths),len(seen),len(cache.expected)), flush=True)
        if seen != set(cache.expected): raise ValueError('Missing training people')
        atomic_json(directory/'complete.json', dict(complete=True, people=len(seen), fingerprint=digest_json(identity)))
        return np.concatenate(xs), np.concatenate(ys)


@torch.inference_mode()
def validate(a, b, base, cache, ann, device, label):
    from predict_a import prediction_records
    from eval_unified import UnifiedPoseEvaluator
    evaluator = UnifiedPoseEvaluator(ann)
    b.eval();seen=set();counts=np.zeros(4,np.int64)
    paths=cache.shards()
    for i,path in enumerate(paths,1):
        arrays=cache.load_shard(path)
        ids=set(map(int,arrays['source_detection_index']))
        if seen & ids:raise ValueError('Duplicate validation IDs')
        seen.update(ids)
        joints=a_choices(a,arrays,device)
        x=torch.as_tensor(make_features(arrays,joints.cpu().numpy()),device=device)
        pred=b(x);alpha=choose_alpha(pred)
        counts+=np.bincount(pred.argmax(1).cpu().numpy(),minlength=4)
        coords,scores=candidate_from_cache(base.head,arrays,joints,alpha,device)
        evaluator.add_records(list(prediction_records(cache,arrays,coords,scores)))
        if i == 1 or i%200==0 or i==len(paths):
            print('%s 验证 [%d/%d] 检测 %d/104125' % (label,i,len(paths),len(seen)),flush=True)
    if seen!=set(cache.expected):raise ValueError('Incomplete validation')
    print(label+' 正在计算完整 COCO AP…',flush=True)
    with tempfile.TemporaryFile(mode='w+') as log,contextlib.redirect_stdout(log):
        metrics,audit=evaluator.evaluate()
    if audit['evaluation_image_count']!=5000 or audit['num_input_instances']!=104125:
        raise ValueError('Incomplete official validation coverage')
    return metrics,audit,{str(alpha):int(n) for alpha,n in zip(ALPHAS,counts)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default=config.DEVICE)
    p.add_argument('--a-checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--labels',type=Path,default=config.WORK/'labels/ab_a3_selected_v1')
    p.add_argument('--check-only',action='store_true',help='Real one-shard interface/label/backprop check; no full training or saved model')
    args=p.parse_args()
    if min(args.epochs,args.batch_size)<1 or not math.isfinite(args.lr) or args.lr<=0 or args.seed<0:
        p.error('Invalid training settings')
    output=(args.output or config.WORK/'runs'/('model_ab_seed%d'%args.seed)).resolve();labeldir=args.labels.resolve()
    protected=[config.TRAIN_CACHE.resolve(),config.VAL_CACHE.resolve(),config.LABELS.resolve()]
    for dest in (output,labeldir):
        if any(dest==q or dest in q.parents or q in dest.parents for q in protected):
            raise ValueError('Output/AB labels must be separate from existing caches and A labels')
    if output==labeldir or output in labeldir.parents or labeldir in output.parents:
        raise ValueError('Keep model output separate from generated labels')
    if output.exists() and not args.check_only:
        raise FileExistsError('模型目录已存在，请用 --output 新目录；原结果不会被覆盖。'+str(output))
    setup(args.seed)
    a_path=args.a_checkpoint or Path(read_result()['experiments'][A_KEY]['checkpoint'])
    saved=torch.load(a_path,map_location='cpu')
    if saved.get('model_config',{}).get('architecture')!='a3_full_joint_context_v1':
        raise ValueError('Expected original A3 architecture')
    if saved.get('base_checkpoint_sha256')!=EXPECTED_CHECKPOINT:raise ValueError('A/base model mismatch')
    a=load_model(saved,args.device).requires_grad_(False).eval()
    cache=ReferenceCache(config.TRAIN_CACHE);require_cache_model(cache)
    val=DetectionCache(config.VAL_CACHE);require_val_cache(val)
    if cache.run['protocol']['dataset_split']!='train2017' or len(cache.expected)!=149813:
        raise ValueError('Require full149813 training cache')
    val_images={int(r['id']) for r in json.loads(config.VAL_ANN.read_text())['images']}
    if len(val_images)!=5000 or set(cache.expected.values())&val_images or set(saved.get('train_image_ids',[]))&val_images:
        raise ValueError('Training/validation overlap or incomplete annotations')
    if hash_file(config.VAL_ANN)!=val.run['protocol']['source_annotation_sha256']:
        raise ValueError('Validation annotation hash mismatch')
    print('固定 A3：%s（第%d轮）；固定CIGPose；只训练B。' % (a_path,saved['epoch']),flush=True)
    from mmengine.logging import MMLogger
    MMLogger.get_current_instance()
    base=load_frozen_base(config.BASE_MODEL,args.device)
    identity=dict(version=ARCHITECTURE,a_sha256=hash_file(a_path),base_sha256=hash_file(config.BASE_MODEL),
                  cache_fingerprint=cache.fingerprint,annotation_sha256=hash_file(cache.directory/'selected_annotations.json'),
                  alphas=list(ALPHAS),p0_reference='original cache P_0_coords',
                  implementation={str(path):hash_file(path) for path in [Path(__file__),Path(__file__).with_name('model_ab.py'),
                    Path(__file__).with_name('model_a3.py'),Path(common.PUBLIC)/'build_shared_cache.py',
                    Path(common.PUBLIC)/'build_a_labels_legacy.py',Path(common.PUBLIC)/'precision_policy.py']})
    if args.check_only:
        arrays=cache.load_shard(cache.shards()[0]);original={k:v.copy() for k,v in arrays.items()}
        ann={int(r['id']):r for r in json.loads((cache.directory/'selected_annotations.json').read_text())['annotations']}
        x,y,joints=label_batch(a,base,arrays,ann,args.device)
        from build_a_labels_legacy import predict_labels
        gt,area=get_gt(arrays,ann);all17,_=predict_labels(base,arrays,gt,area,args.device)
        np.testing.assert_allclose(y[:,3],all17['gain'][np.arange(len(joints)),joints],rtol=0,atol=1e-8)
        b=BRegressor().to(args.device);b.set_statistics(x)
        b_before={k:v.detach().clone() for k,v in b.named_parameters()}
        before={k:v.detach().clone() for k,v in a.state_dict().items()}
        base_before={k:v.detach().clone() for k,v in base.state_dict().items()}
        opt=torch.optim.AdamW(b.parameters(),lr=args.lr)
        pred=b(torch.as_tensor(x,device=args.device));loss=torch.nn.functional.smooth_l1_loss(pred,torch.as_tensor(y,device=args.device)*GAIN_SCALE)
        loss.backward();opt.step()
        assert torch.isfinite(loss) and all(v.grad is None for v in a.parameters()) and all(v.grad is None for v in base.parameters())
        assert all(torch.equal(v,before[k]) for k,v in a.state_dict().items())
        assert all(torch.equal(v,base_before[k]) for k,v in base.state_dict().items())
        assert any(not torch.equal(v,b_before[k]) for k,v in b.named_parameters())
        assert all(np.array_equal(v,original[k]) for k,v in arrays.items())
        assert choose_alpha(torch.zeros(2,4,device=args.device)).tolist()==[.25,.25]
        assert choose_alpha(torch.tensor([[-4.,-3.,-1.,-2.]],device=args.device)).item()==.75
        altered=(joints+1)%17;alternative=make_features(arrays,altered)
        np.testing.assert_array_equal(alternative[:,:256],arrays['features'][np.arange(len(altered)),altered])
        np.testing.assert_array_equal(alternative[:,-17:],np.eye(17,dtype=np.float32)[altered])
        # Exercise interrupted-label reuse and checksum rejection on real data.
        from types import SimpleNamespace
        one=cache.shards()[0]
        small=SimpleNamespace(directory=cache.directory,shards=lambda:[one],load_shard=cache.load_shard,
                              expected={int(i):int(im) for i,im in zip(arrays['annotation_id'],arrays['image_id'])})
        with tempfile.TemporaryDirectory(prefix='ab-label-check-') as temp:
            directory=Path(temp)
            x1,y1=prepare_labels(a,base,small,directory,identity,args.device)
            x2,y2=prepare_labels(a,base,small,directory,identity,args.device)
            np.testing.assert_array_equal(x1,x2);np.testing.assert_array_equal(y1,y2)
            receipt=(directory/one.name).with_suffix('.json')
            bad=json.loads(receipt.read_text());bad['payload_sha256']='invalid';atomic_json(receipt,bad)
            try:prepare_labels(a,base,small,directory,identity,args.device)
            except ValueError as error:
                if 'checksum mismatch' not in str(error):raise
            else:raise AssertionError('Corrupted label checksum was accepted')
        print('PASS：真实%d人；A选关节四强度标签；alpha=1与原A标签一致；B反向更新；A/CIGPose权重和缓存不变。未执行完整训练或AP验证。'%len(joints),flush=True)
        return
    print('检查完整训练/验证缓存…',flush=True);cache.verify(True);val.verify(True)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'writer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        x,y=prepare_labels(a,base,cache,labeldir,identity,args.device)
        setup(args.seed)  # Reset after label generation, so reused labels do not change initialization.
        b=BRegressor().to(args.device);b.set_statistics(x)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)*GAIN_SCALE),
                    batch_size=args.batch_size,shuffle=True,num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed))
        optimizer=torch.optim.AdamW(b.parameters(),lr=args.lr,weight_decay=.0001)
        settings=dict(identity=identity,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,seed=args.seed,
                      train_people=len(x),gain_scale=GAIN_SCALE,architecture=ARCHITECTURE,
                      b_source_commit='f1df4442b02eb14340dbbde4bc8d5cb4aa3a480e',
                      a_checkpoint=str(a_path),val_cache_fingerprint=val.fingerprint,
                      description='冻结A3和CIGPose，只训练B；A选关节，B选四强度，始终接受；原缓存P0为标签基准。')
        atomic_json(output/'settings.json',settings)
        history=[];best=-float('inf');best_epoch=None
        for epoch in range(1,args.epochs+1):
            started=time.monotonic();b.train();total=0.;count=0
            label='Epoch %d/%d'%(epoch,args.epochs)
            for i,(bx,by) in enumerate(loader,1):
                bx=bx.to(args.device);by=by.to(args.device);optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.smooth_l1_loss(b(bx),by)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(b.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                count+=len(bx);total+=float(loss.detach())*len(bx)
                if i==1 or i%50==0 or i==len(loader):
                    print('%s 训练 [%d/%d] 人体 %d/%d Loss=%.6f'%(label,i,len(loader),count,len(x),total/count),flush=True)
            assert count==149813
            metrics,audit,counts=validate(a,b,base,val,config.VAL_ANN,args.device,label)
            ap=metrics['coco/AP'];improved=ap>best
            if improved:
                best=ap;best_epoch=epoch
                payload=dict(architecture=ARCHITECTURE,model_state={k:v.detach().cpu().clone() for k,v in b.state_dict().items()},
                             epoch=epoch,validation=metrics,settings=settings,alphas=list(ALPHAS),trained=True)
                tmp=output/'best.pt.tmp';torch.save(payload,tmp);tmp.replace(output/'best.pt')
            row=dict(epoch=epoch,count=count,loss=total/count,validation=metrics,audit=audit,
                     alpha_counts=counts,seconds=time.monotonic()-started)
            history.append(row)
            update_experiment(output.name,dict(title='A3固定＋B强度选择',description=settings['description'],settings=settings,
                              history=history,best_epoch=best_epoch,checkpoint=str(output/'best.pt')))
            print('%s 完成：Loss=%.6f AP=%.6f 最佳=第%d轮 %.6f%s'%(label,total/count,ap*100,best_epoch,best*100,' [新最佳]' if improved else ''),flush=True)
        best_payload=torch.load(output/'best.pt',map_location='cpu');b.load_state_dict(best_payload['model_state'])
        checked,audit,counts=validate(a,b,base,val,config.VAL_ANN,args.device,'最佳模型复核')
        for key,value in best_payload['validation'].items():
            if abs(checked[key]-value)>1e-12:raise ValueError('Best checkpoint validation mismatch: '+key)
        if hash_file(a_path)!=identity['a_sha256']:raise ValueError('A checkpoint changed during run')
        record=dict(title='A3固定＋B强度选择',description=settings['description'],settings=settings,history=history,
                    best_epoch=best_epoch,checkpoint=str(output/'best.pt'),completed=True,best_recheck=checked)
        update_experiment(output.name,record)
        print('\n最终结果（AP/AR为0–100）\n轮次     Loss         AP         AR     标记',flush=True)
        for r in history:
            print('%3d   %.6f   %.6f   %.6f   %s'%(r['epoch'],r['loss'],r['validation']['coco/AP']*100,
                  r['validation']['coco/AR']*100,'最终最佳' if r['epoch']==best_epoch else ''),flush=True)
        print('最佳：第%d轮，AP=%.6f；已重新加载并完整复核。\n模型：%s\n统一结果：%s'%(best_epoch,best*100,output/'best.pt',config.WORK/'code/experiments/module_a/results.html'),flush=True)
        print('最终最佳模型完整指标：',flush=True)
        for key,value in checked.items():print('  %s = %.6f'%(key,value*100),flush=True)


if __name__=='__main__':main()
