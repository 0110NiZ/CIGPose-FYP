"""Regenerate A's 17 signed gains from existing train GT features; never train.

Same intervention/OKS definition as Y's generator. Uses the approved legacy
public candidate decoder. Labels stay incomplete until verify_a_labels_legacy.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch

from a_contract import label_metadata_template, validate_label_arrays
from a_runtime import setup, load_frozen_base, require_cache_model
from build_shared_cache import candidate_from_cache
from precision_policy import runtime_record, assert_legacy_precision
from shared_cache import ReferenceCache, hash_file, atomic_json, digest_json
from oks_utility import COCO_SIGMAS, compute_instance_oks


def batch_oks(coords, gt, area):
    """Vectorized shared OKS expression; checked against scalar public function."""
    coords=np.asarray(coords,dtype=np.float64)
    gt=np.asarray(gt,dtype=np.float64);area=np.asarray(area,dtype=np.float64)
    visible=gt[:,:,2]>0
    if not visible.any(1).all() or not np.isfinite(coords).all() or not (area>0).all():
        raise ValueError('Invalid OKS batch')
    d=coords-gt[:,:,:2]
    exponent=(d*d).sum(-1)/(2*COCO_SIGMAS)**2/(area[:,None]+np.spacing(1))/2
    return (np.exp(-exponent)*visible).sum(1)/visible.sum(1)


def get_gt(arrays, annotations):
    selected=[annotations[int(i)] for i in arrays['annotation_id']]
    if any(int(a['image_id'])!=int(i) for a,i in zip(selected,arrays['image_id'])):
        raise ValueError('GT image/person mismatch')
    return np.asarray([a['keypoints'] for a in selected],np.float64).reshape(-1,17,3),np.asarray([a['area'] for a in selected],np.float64)


def predict_labels(base, arrays, gt, area, device):
    assert_legacy_precision()
    if base.training or any(p.requires_grad for p in base.parameters()):
        raise ValueError('Base model must stay frozen eval')
    n=len(arrays['features']); original=arrays['features'].copy()
    f=torch.as_tensor(arrays['features'],device=device)
    ids=torch.zeros(n,dtype=torch.long,device=device)
    zeros=torch.zeros(n,device=device)
    unmodified,mask=base.head.cim.apply_intervention(f,ids,zeros)
    if not torch.equal(unmodified,f) or mask.any():
        raise ValueError('Alpha=0 changed features')
    p0,score0=candidate_from_cache(base.head,arrays,ids,zeros,device)
    before=batch_oks(arrays['P_0_coords'],gt,area)
    rebuilt_before=batch_oks(p0,gt,area)
    after=[]
    for joint in range(17):
        joints=torch.full((n,),joint,dtype=torch.long,device=device)
        changed,mask=base.head.cim.apply_intervention(f,joints,torch.ones(n,device=device))
        if int(mask.sum())!=n or not mask[:,joint].all():
            raise ValueError('Incorrect intervention mask')
        if not torch.equal(changed[:,joint],base.head.cim.canonical_features.weight[joint].expand(n,-1)):
            raise ValueError('Alpha=1 not canonical replacement')
        # Public entry always reloads untouched features for each candidate.
        coords,_=candidate_from_cache(base.head,arrays,joints,torch.ones(n,device=device),device)
        after.append(batch_oks(coords,gt,area))
    if not np.array_equal(original,arrays['features']) or not np.array_equal(f.cpu().numpy(),original):
        raise ValueError('Input feature mutation')
    after=np.stack(after,axis=1)
    labels=dict(annotation_id=arrays['annotation_id'],image_id=arrays['image_id'],oks_before=before,oks_after=after,gain=after-before[:,None])
    validate_label_arrays(labels)
    diagnostics=dict(rebuilt_before=rebuilt_before,
        p0_coordinate_error=np.abs(p0-arrays['P_0_coords']).max(axis=(1,2)),
        p0_score_error=np.abs(score0-arrays['P_0_scores']).max(axis=1))
    return labels,diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--y-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); runtime=setup()
    cache=ReferenceCache(args.cache);require_cache_model(cache)
    cache.verify(True)
    out=args.output.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('Choose a new label version or use --resume')
    out.mkdir(parents=True,exist_ok=True)
    protocol=label_metadata_template(cache)
    protocol.update(precision_profile=runtime['precision_profile'],precision_profile_sha256=runtime['precision_profile_sha256'],
        version='module-a-train2017-legacy-deterministic-v2',generator_source_snapshot='source/build_a_labels_legacy.py',
        generator_source_sha256=hash_file(__file__),oks_implementation_sha256=hash_file(Path(__file__).with_name('oks_utility.py')),
        original_y_archive_sha256=hash_file(args.y_archive),p0_reference='unchanged source cache P_0_coords',
        source_note='Regenerated locally using the shared Y intervention/gain definition; archived source hashes identify uncommitted generator exactly.')
    source_files=['build_a_labels_legacy.py','verify_a_labels_legacy.py','precision_policy.py','a_contract.py','a_runtime.py','build_shared_cache.py','shared_cache.py','oks_utility.py']
    hashes={n:hash_file(Path(__file__).with_name(n)) for n in source_files}
    protocol['implementation_sha256']=hashes
    identity=digest_json(protocol)
    status_path=out/'generation_status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text())
        if status['identity']!=identity or json.loads((out/'metadata.json').read_text())['complete']:
            raise ValueError('Resume source/settings changed or labels already finalized')
    else:
        status=dict(identity=identity,done={},complete=False)
        (out/'source').mkdir(exist_ok=True)
        for n in source_files:shutil.copyfile(Path(__file__).with_name(n),out/'source'/n)
        atomic_json(out/'metadata.json',protocol)
        atomic_json(out/'runtime.json',runtime)
    annotations={int(a['id']):a for a in json.loads((cache.directory/'selected_annotations.json').read_text())['annotations']}
    archive=zipfile.ZipFile(args.y_archive);ystatus=json.loads(archive.read('train_a_labels/status.json'))
    base=load_frozen_base(args.checkpoint,args.device)
    start=time.time();paths=cache.shards()
    with torch.inference_mode():
        for index,path in enumerate(paths):
            name='shard-%05d.npz'%index
            if path.name in status['done']:
                item=status['done'][path.name]
                if hash_file(out/item['file'])!=item['sha256']:raise ValueError('Resume checksum mismatch')
                continue
            arrays=cache.load_shard(path);gt,area=get_gt(arrays,annotations)
            labels,diag=predict_labels(base,arrays,gt,area,args.device)
            reference=np.array([compute_instance_oks(arrays['P_0_coords'][i],gt[i],area[i]) for i in range(len(gt))])
            if not np.allclose(reference,labels['oks_before'],atol=1e-14,rtol=0):raise ValueError('Batch/scalar OKS mismatch')
            yentry=ystatus['done'][path.name];raw=archive.read('train_a_labels/'+yentry['file'])
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                if not np.array_equal(z['annotation_id'],arrays['annotation_id']):raise ValueError('Y ID mismatch')
                yg=z['gain'].copy()
            with (out/(name+'.tmp')).open('wb') as f:np.savez_compressed(f,**labels)
            (out/(name+'.tmp')).replace(out/name)
            p0diff=np.abs(diag['rebuilt_before']-labels['oks_before'])
            gain_error=np.abs(labels['gain']-yg)
            item=dict(file=name,sha256=hash_file(out/name),count=len(gt),source=path.name,
                p0_max_coordinate_error=float(diag['p0_coordinate_error'].max()),p0_max_oks_error=float(p0diff.max()),
                p0_oks_errors_gt_1e5=int((p0diff>1e-5).sum()),p0_max_score_error=float(diag['p0_score_error'].max()),
                p0_worst_annotation_id=int(arrays['annotation_id'][p0diff.argmax()]),
                y_max_gain_difference=float(gain_error.max()),y_gain_differences_gt_1e5=int((gain_error>1e-5).sum()),
                y_best_joint_changes=int((labels['gain'].argmax(1)!=yg.argmax(1)).sum()))
            status['done'][path.name]=item
            atomic_json(status_path,status)
            if (index+1)%100==0 or index+1==len(paths):
                print(json.dumps(dict(shards=index+1,total_shards=len(paths),persons=sum(v['count'] for v in status['done'].values()),elapsed_seconds=time.time()-start)),flush=True)
    status['complete']=True
    atomic_json(status_path,status)
    atomic_json(out/'generation_summary.json',dict(generation_complete=True,accepted=False,training=False,
        shards=len(status['done']),persons=sum(v['count'] for v in status['done'].values()),
        max_p0_coordinate_error=max(v['p0_max_coordinate_error'] for v in status['done'].values()),
        max_p0_oks_error=max(v['p0_max_oks_error'] for v in status['done'].values()),
        p0_oks_errors_gt_1e5=sum(v['p0_oks_errors_gt_1e5'] for v in status['done'].values()),
        y_gain_differences_gt_1e5=sum(v['y_gain_differences_gt_1e5'] for v in status['done'].values()),
        y_best_joint_changes=sum(v['y_best_joint_changes'] for v in status['done'].values())))
    print('Generation complete. Metadata remains incomplete pending independent verification.',flush=True)


if __name__=='__main__':main()
