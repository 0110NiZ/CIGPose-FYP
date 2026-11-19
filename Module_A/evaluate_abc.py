"""Frozen A+B+C transfer evaluation; C must receive the actual AB candidate."""
import argparse
import contextlib
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import common
import config
from a_runtime import setup,load_frozen_base,require_val_cache
from detection_cache import ReferenceCache
from shared_cache import hash_file,atomic_json
from evaluate import load_model
from model_ab import BRegressor,make_features,choose_alpha,ARCHITECTURE
from train_ab import a_choices
from model_abc import ModuleCAcceptanceClassifier,encode_module_c_features
from build_shared_cache import candidate_from_cache
from predict_a import prediction_records
from eval_unified import UnifiedPoseEvaluator
from result import read_result,update_experiment


def c_inputs(arrays,coords,scores,joints,alpha,device):
    def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=device)
    x=encode_module_c_features(tensor(arrays['P_0_coords']),tensor(coords),tensor(arrays['bbox']),
        tensor(arrays['P_0_scores']),tensor(scores),joints,alpha,tensor(arrays['uncertainty']))
    if x.shape!=(len(joints),137) or not torch.isfinite(x).all():raise ValueError('Invalid C features')
    return x


def fallback(arrays,coords,scores,probability,tau):
    p=np.asarray(probability)
    if p.shape!=(len(coords),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid C probabilities')
    accept=p>tau
    return (np.where(accept[:,None,None],coords,arrays['P_0_coords']),
            np.where(accept[:,None],scores,arrays['P_0_scores']),accept)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c-checkpoint',type=Path)
    p.add_argument('--ab-checkpoint',type=Path,default=config.WORK/'runs/model_ab_seed42/best.pt')
    p.add_argument('--a-checkpoint',type=Path)
    p.add_argument('--tau',type=float,default=.51)
    p.add_argument('--device',default=config.DEVICE)
    p.add_argument('--output',type=Path,default=config.WORK/'runs/evaluate_abc_frozen')
    p.add_argument('--check-only',action='store_true',help='One real batch; without C weights uses a random C for interface checks ONLY')
    args=p.parse_args()
    if not 0<=args.tau<=1:p.error('tau must be in [0,1]')
    if not args.check_only and (args.c_checkpoint is None or not args.c_checkpoint.is_file()):
        p.error('完整验证需要真实 C 权重：--c-checkpoint /path/to/epoch_014.pt')
    if not args.check_only and args.output.exists():raise FileExistsError('Choose a new --output')
    setup(42)
    ab=torch.load(args.ab_checkpoint,map_location='cpu')
    if ab.get('architecture')!=ARCHITECTURE or not ab.get('trained'):raise ValueError('Expected trained AB B checkpoint')
    a_path=args.a_checkpoint or Path(ab['settings']['a_checkpoint'])
    if hash_file(a_path)!=ab['settings']['identity']['a_sha256']:raise ValueError('A differs from the A used to train B')
    a=load_model(torch.load(a_path,map_location='cpu'),args.device).requires_grad_(False).eval()
    b=BRegressor().to(args.device);b.load_state_dict(ab['model_state'],strict=True);b.requires_grad_(False).eval()
    if args.c_checkpoint:
        saved=torch.load(args.c_checkpoint,map_location='cpu')
        if saved.get('input_dim')!=137:raise ValueError('Require original137d C, not145d/multitask variants')
        c=ModuleCAcceptanceClassifier(saved['input_dim'],saved['hidden_dim'],saved['dropout']).to(args.device)
        c.load_state_dict(saved['model_state'],strict=True)
        c_info=dict(path=str(args.c_checkpoint),sha256=hash_file(args.c_checkpoint),epoch=saved['epoch'],training=saved.get('training'))
    else:
        c=ModuleCAcceptanceClassifier(137,128,.1).to(args.device);c_info=dict(random_interface_check_only=True)
    c.requires_grad_(False).eval()
    cache=ReferenceCache(config.VAL_CACHE);require_val_cache(cache)
    if hash_file(config.VAL_ANN)!=cache.run['protocol']['source_annotation_sha256']:raise ValueError('Annotation mismatch')
    base=load_frozen_base(config.BASE_MODEL,args.device)
    if hash_file(config.BASE_MODEL)!=ab['settings']['identity']['base_sha256']:raise ValueError('Base checkpoint mismatch')
    paths=cache.shards()
    if args.check_only:paths=paths[:1]
    else:
        cache.verify(True);args.output.mkdir(parents=True)
    evaluators={} if args.check_only else {key:UnifiedPoseEvaluator(config.VAL_ANN) for key in ['P0','AB','ABC']}
    seen=set();accepted=0
    with torch.inference_mode(),contextlib.ExitStack() as stack:
        decisions=None if args.check_only else stack.enter_context((args.output/'decisions.jsonl').open('w'))
        for seq,path in enumerate(paths,1):
            ar=cache.load_shard(path);j=a_choices(a,ar,args.device)
            alpha=choose_alpha(b(torch.as_tensor(make_features(ar,j.cpu().numpy()),device=args.device)))
            pc,sc=candidate_from_cache(base.head,ar,j,alpha,args.device)
            x=c_inputs(ar,pc,sc,j,alpha,args.device);prob=c.predict_proba(x).cpu().numpy()
            final,fs,mask=fallback(ar,pc,sc,prob,args.tau)
            if args.check_only:
                keep,ks,_=fallback(ar,pc,sc,np.ones(len(j)),.51)
                reject,rs,_=fallback(ar,pc,sc,np.zeros(len(j)),.51)
                assert np.array_equal(keep,pc) and np.array_equal(ks,sc)
                assert np.array_equal(reject,ar['P_0_coords']) and np.array_equal(rs,ar['P_0_scores'])
                assert not fallback(ar,pc,sc,np.full(len(j),.51),.51)[2].any()
                np.testing.assert_allclose(x[:,119].cpu().numpy(),alpha.cpu().numpy(),rtol=0,atol=0)
                np.testing.assert_array_equal(x[:,102:119].argmax(1).cpu().numpy(),j.cpu().numpy())
                print('PASS: real %d detections; A/B candidate -> C137d; actual joint/alpha encoded; accept/reject/tie fallback verified. C=%s. NO AP RESULT.'%(len(j),c_info),flush=True)
                return
            ids=set(map(int,ar['source_detection_index']))
            if seen&ids:raise ValueError('Duplicate detections')
            seen.update(ids);accepted+=int(mask.sum())
            for key,coords,scores in [('P0',ar['P_0_coords'],ar['P_0_scores']),('AB',pc,sc),('ABC',final,fs)]:
                evaluators[key].add_records(list(prediction_records(cache,ar,coords,scores)))
            for idx,joint,strength,pr,ok in zip(ar['source_detection_index'],j.cpu().tolist(),alpha.cpu().tolist(),prob,mask):
                decisions.write(json.dumps(dict(source_detection_index=int(idx),joint=joint,alpha=strength,p_acc=float(pr),accept=bool(ok)))+'\n')
            if seq==1 or seq%200==0 or seq==len(paths):print('ABC [%d/%d] %d/104125 detections; accept %.2f%%'%(seq,len(paths),len(seen),100*accepted/len(seen)),flush=True)
    if seen!=set(cache.expected):raise ValueError('Incomplete detection coverage')
    results={}
    for key,evaluator in evaluators.items():
        print('Scoring '+key,flush=True)
        with tempfile.TemporaryFile(mode='w+') as log,contextlib.redirect_stdout(log):metrics,audit=evaluator.evaluate()
        assert audit['evaluation_image_count']==5000 and audit['num_input_instances']==104125
        results[key]=dict(metrics=metrics,audit=audit)
    for key,value in ab['validation'].items():
        if abs(results['AB']['metrics'][key]-value)>1e-12:raise ValueError('AB checkpoint parity failed: '+key)
    record=dict(method='frozen A3 + AB-trained B + transferred C; no retraining or threshold sweep',
        a_checkpoint=str(a_path),a_sha256=hash_file(a_path),ab_checkpoint=str(args.ab_checkpoint),ab_sha256=hash_file(args.ab_checkpoint),
        c=c_info,tau=args.tau,accept_rate=accepted/len(seen),count=len(seen),results=results,
        caveat='C trained on Top1 alpha1, now applied to A-selected variable-alpha candidates; validation not independent test')
    atomic_json(args.output/'results.json',record)
    update_experiment(args.output.name,dict(title='A+B+C 固定模型串联',description=record['method'],settings=record,
        checkpoint=str(args.c_checkpoint),history=[dict(epoch=c_info['epoch'],validation=results['ABC']['metrics'])],
        comparison_only=True,completed=True))
    print('\n最终比较（AP 0–100）：',flush=True)
    for key,value in results.items():print('%s: %.6f'%(key,value['metrics']['coco/AP']*100),flush=True)
    print('ABC-AB: %+.6f AP点；接受率 %.2f%%；结果 %s'%((results['ABC']['metrics']['coco/AP']-results['AB']['metrics']['coco/AP'])*100,100*accepted/len(seen),args.output/'results.json'),flush=True)


if __name__=='__main__':main()
