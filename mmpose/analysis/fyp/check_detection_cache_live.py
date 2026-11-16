"""Recompute first/middle/last committed batches and compare with full cache."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
from build_detection_cache import ROOT, DetectionDataset
from build_shared_cache import array, decode_original, candidate_from_cache
from detection_cache import ReferenceCache
from shared_cache import atomic_json
from read_val_detection_boxes import load_detection_instances


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cache=ReferenceCache(a.cache)
    _,records=load_detection_instances(a.data/'person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        a.data/'annotations/person_keypoints_val2017.json',a.data/'val2017')
    sys.path.insert(0,str(ROOT/'mmpose'));os.chdir(ROOT/'mmpose')
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmpose.apis import init_model
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    from mmpose.utils import register_all_modules
    register_all_modules()
    from precision_policy import configure_legacy_precision
    configure_legacy_precision()
    cfg=Config.fromfile(str(a.cache/'resolved_config.py'))
    model=init_model(cfg,str(a.checkpoint),device='cuda:0');model.requires_grad_(False).eval()
    meta=parse_pose_metainfo(dict(from_file='configs/_base_/datasets/coco.py'))
    paths=cache.shards();chosen=[paths[0],paths[len(paths)//2],paths[-1]]
    report=[]
    with torch.inference_mode():
        for path in chosen:
            stored=cache.load_shard(path)
            subset=[records[int(i)] for i in stored['source_detection_index']]
            ds=DetectionDataset(subset,cfg.val_pipeline,meta)
            batch=pseudo_collate([ds[i] for i in range(len(ds))])
            processed=model.data_preprocessor(batch,training=False);samples=processed['data_samples']
            feats=model.extract_feat(processed['inputs'])
            original=model.add_pred_to_datasample(model.head.predict(feats,samples,model.test_cfg),None,deepcopy(samples))
            state=model.head.extract_refinement_state(feats)
            for field in ('features','uncertainty','confidence'):
                np.testing.assert_allclose(array(state[field]),stored[field],atol=1e-5,rtol=1e-5)
            np.testing.assert_array_equal(array(state['top1_indices']),stored['top1_joint'])
            p0=decode_original(model,state['p0_simcc'],samples)
            # Separate exact cache/reload behavior from cross-process GPU recomputation.
            fresh=dict(stored)
            fresh['features']=array(state['features']).copy()
            fresh['P_0_coords']=np.stack([array(r.pred_instances.keypoints)[0] for r in p0])
            fresh['P_0_scores']=np.stack([array(r.pred_instances.keypoint_scores)[0] for r in p0])
            reloaded=torch.as_tensor(fresh['features'],device='cuda:0')
            assert torch.equal(reloaded,state['features'])
            joints=torch.arange(len(samples),device='cuda:0')%17
            alphas=torch.tensor([.25,.5,.75,1.],device='cuda:0')[torch.arange(len(samples),device='cuda:0')%4]
            reload_checks=[]
            for action in (torch.zeros(len(samples),device='cuda:0'),alphas):
                changed,_=model.head.cim.apply_intervention(state['features'],joints,action)
                changed_reload,_=model.head.cim.apply_intervention(reloaded,joints,action)
                assert torch.equal(changed,changed_reload)
                logits=model.head.predict_from_joint_features(changed)
                logits_reload=model.head.predict_from_joint_features(changed_reload)
                logit_delta=max(float((x-y).abs().max()) for x,y in zip(logits,logits_reload))
                assert logit_delta<1e-4
                live=decode_original(model,logits,samples)
                c,sc=candidate_from_cache(model.head,fresh,joints,action,'cuda:0')
                live_c=np.stack([array(r.pred_instances.keypoints)[0] for r in live])
                live_sc=np.stack([array(r.pred_instances.keypoint_scores)[0] for r in live])
                np.testing.assert_allclose(c,live_c,atol=.1,rtol=0)
                np.testing.assert_allclose(sc,live_sc,atol=1e-5,rtol=1e-6)
                if not action.any():
                    np.testing.assert_allclose(c,fresh['P_0_coords'],atol=.1,rtol=0)
                    np.testing.assert_allclose(sc,fresh['P_0_scores'],atol=1e-5,rtol=1e-6)
                reload_checks.append(dict(equal_feature_values=True,logit_max_abs_difference=logit_delta,
                    coordinate_max_abs_difference=float(np.abs(c-live_c).max()),
                    score_max_abs_difference=float(np.abs(sc-live_sc).max())))
            delta=0.; score_delta=0.
            feature_delta=float(np.max(np.abs(array(state['features'])-stored['features'])))
            for prefix,results in [('P_0',p0),('original',original)]:
                coords=np.stack([array(r.pred_instances.keypoints)[0] for r in results])
                scores=np.stack([array(r.pred_instances.keypoint_scores)[0] for r in results])
                np.testing.assert_allclose(coords,stored[prefix+'_coords'],atol=.1,rtol=0)
                np.testing.assert_allclose(scores,stored[prefix+'_scores'],atol=1e-5,rtol=1e-6)
                np.testing.assert_array_equal(scores>.2,stored[prefix+'_scores']>.2)
                score_delta=max(score_delta,float(np.max(np.abs(scores-stored[prefix+'_scores']))))
                delta=max(delta,float(np.max(np.abs(coords-stored[prefix+'_coords']))))
            report.append(dict(shard=path.name,count=len(ds),max_coordinate_abs_difference=delta,
                               max_score_abs_difference=score_delta,max_feature_abs_difference=feature_delta,reload_checks=reload_checks))
    atomic_json(a.output,dict(passed=True,batches=report,checked_count=sum(r['count'] for r in report),
                             cross_process_tolerance=dict(feature_atol=1e-5,feature_rtol=1e-5,coordinate_atol=.1,coordinate_rtol=0,score_atol=1e-5,score_rtol=1e-6),
                             note='Cache bytes roundtrip exactly. GPU recomputation is checked within explicit float32 tolerance; decoded argmax is not bitwise reproducible. No scoring-threshold or Top-1 changes permitted in sampled comparisons.',
                             checks=['live features','uncertainty','top1','P0/original coords and scores','alpha0','mixed candidate reload']))
    print('PASS',report)


if __name__=='__main__':main()
