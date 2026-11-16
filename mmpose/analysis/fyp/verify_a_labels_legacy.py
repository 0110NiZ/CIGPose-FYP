"""Independent full file/GT audit plus selected GPU recomputation; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from a_contract import GainLabels, validate_label_arrays, label_metadata_template
from a_runtime import setup, load_frozen_base, require_cache_model
from precision_policy import require_profile, runtime_record
from shared_cache import ReferenceCache, hash_file, atomic_json
from build_shared_cache import candidate_from_cache
from oks_utility import compute_instance_oks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--finalize',action='store_true')
    args=p.parse_args();setup()
    cache=ReferenceCache(args.cache);require_cache_model(cache)
    out=args.labels.resolve();meta=json.loads((out/'metadata.json').read_text())
    require_profile(meta['precision_profile'],meta['precision_profile_sha256'])
    expected=label_metadata_template(cache)
    for k in ['source_cache_fingerprint','checkpoint_sha256','annotation_sha256','joint_order','alpha','flip_test','gain_definition']:
        if meta[k]!=expected[k]:raise ValueError('Label protocol mismatch: '+k)
    for name,sha in meta['implementation_sha256'].items():
        if hash_file(out/'source'/name)!=sha:raise ValueError('Source snapshot mismatch')
    status=json.loads((out/'generation_status.json').read_text())
    if not status['complete'] or set(status['done'])!={p.name for p in cache.shards()}:
        raise ValueError('Incomplete source coverage')
    if {p.name for p in out.glob('*.npz')}!={e['file'] for e in status['done'].values()}:
        raise ValueError('Shard manifest mismatch')
    annotations={int(a['id']):a for a in json.loads((cache.directory/'selected_annotations.json').read_text())['annotations']}
    seen=set();chunks=[];max_before_error=0.;manifest=[]
    for source,e in sorted(status['done'].items()):
        file=out/e['file']
        if hash_file(file)!=e['sha256']:raise ValueError('Label checksum mismatch')
        with np.load(file,allow_pickle=False) as z:lab={k:z[k] for k in z.files}
        validate_label_arrays(lab)
        a=cache.load_shard(cache.directory/source)
        ids=lab['annotation_id'].tolist()
        if seen.intersection(ids) or len(ids)!=e['count']:raise ValueError('Duplicate or wrong label count')
        if not np.array_equal(lab['annotation_id'],a['annotation_id']) or not np.array_equal(lab['image_id'],a['image_id']):
            raise ValueError('ID correspondence mismatch')
        seen.update(ids)
        before=[]
        for i,aid in enumerate(ids):
            ann=annotations[aid]
            if ann['image_id']!=int(lab['image_id'][i]):raise ValueError('Wrong GT image')
            before.append(compute_instance_oks(a['P_0_coords'][i],np.asarray(ann['keypoints']).reshape(17,3),ann['area']))
        error=float(np.max(np.abs(np.array(before)-lab['oks_before'])))
        if error>1e-12:raise ValueError('Independent scalar OKS-before mismatch')
        max_before_error=max(max_before_error,error)
        chunks.append(lab['gain']);manifest.append({k:e[k] for k in ['file','sha256','count']})
    if seen!=set(cache.expected):raise ValueError('Missing people')
    names=sorted(status['done']);chosen={names[0],names[len(names)//2],names[-1]}
    chosen.update(names[i] for i in np.linspace(0,len(names)-1,32,dtype=int))
    chosen.update(names[i] for i in np.random.RandomState(20261116).choice(len(names),32,replace=False))
    for key in ['p0_max_oks_error','p0_max_coordinate_error','y_max_gain_difference','y_best_joint_changes']:
        chosen.update(sorted(names,key=lambda n:status['done'][n][key],reverse=True)[:8])
    base=load_frozen_base(args.checkpoint,args.device)
    checks=[]
    with torch.inference_mode():
        for source in sorted(chosen):
            a=cache.load_shard(cache.directory/source);n=len(a['features'])
            with np.load(out/status['done'][source]['file'],allow_pickle=False) as z:saved=z['oks_after'].copy()
            after=[]
            for j in range(17):
                coords,_=candidate_from_cache(base.head,a,torch.full((n,),j,dtype=torch.long,device=args.device),torch.ones(n,device=args.device),args.device)
                values=[]
                for i,aid in enumerate(a['annotation_id']):
                    ann=annotations[int(aid)]
                    values.append(compute_instance_oks(coords[i],np.asarray(ann['keypoints']).reshape(17,3),ann['area']))
                after.append(values)
            after=np.stack(after,axis=1);error=np.abs(after-saved)
            checks.append(dict(source=source,persons=n,max_after_error=float(error.max()),
                errors_gt_1e5=int((error>1e-5).sum()),best_joint_changes=int((after.argmax(1)!=saved.argmax(1)).sum())))
    gains=np.concatenate(chunks);repeat_pass=all(r['max_after_error']<=1e-5 and r['best_joint_changes']==0 for r in checks)
    report=dict(passed=repeat_pass,full_file_and_scalar_gt_audit=True,persons=len(seen),shards=len(manifest),gain_values=int(gains.size),
        max_scalar_before_error=max_before_error,mean_gain=float(gains.mean()),mean_oracle_gain=float(gains.max(1).mean()),
        gain_positive=int((gains>0).sum()),gain_negative=int((gains<0).sum()),gain_zero=int((gains==0).sum()),
        gpu_recheck_shards=len(checks),gpu_recheck_persons=sum(c['persons'] for c in checks),gpu_recheck_candidate_count=sum(c['persons']*17 for c in checks),
        gpu_recheck_max_after_error=max(c['max_after_error'] for c in checks),gpu_recheck_best_joint_changes=sum(c['best_joint_changes'] for c in checks),
        gpu_recheck_tolerance=1e-5,gpu_checks=checks,training=False,optimizer_steps=0,precision=runtime_record(),
        p0_reference_note='Labels subtract unchanged source-cache P0; reconstructed P0 differences are reported, not asserted zero.',
        generation_summary=json.loads((out/'generation_summary.json').read_text()))
    atomic_json(out/'acceptance_report.json',report)
    if not repeat_pass:raise RuntimeError('Candidate recomputation failed; metadata not finalized')
    if args.finalize:
        meta.update(count=len(seen),complete=True,shards=sorted(manifest,key=lambda x:x['file']),
                    acceptance_report_sha256=hash_file(out/'acceptance_report.json'))
        # Keep the pre-finalization manifest as provenance.
        if not (out/'metadata.before_acceptance.json').exists():
            atomic_json(out/'metadata.before_acceptance.json',json.loads((out/'metadata.json').read_text()))
        atomic_json(out/'metadata.json',meta)
        try:
            labels=GainLabels(out,cache)
            if len(labels.ids)!=len(seen):raise ValueError('Reader omitted labels')
        except Exception:
            meta['complete']=False;atomic_json(out/'metadata.json',meta)
            raise
        atomic_json(out/'reader_check.json',dict(passed=True,count=len(labels.ids),fingerprint=labels.fingerprint,training=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ['gpu_checks','precision']},indent=2),flush=True)


if __name__=='__main__':main()
