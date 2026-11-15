"""Recompute first/middle/last committed batches and compare with full cache."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
from build_detection_cache import ROOT, DetectionDataset
from build_shared_cache import array, decode_original, check_reloaded
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
    register_all_modules();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
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
            feats=model.extract_feat(processed['inputs']);state=model.head.extract_refinement_state(feats)
            for field in ('features','uncertainty','confidence'):
                np.testing.assert_allclose(array(state[field]),stored[field],atol=1e-5,rtol=1e-5)
            np.testing.assert_array_equal(array(state['top1_indices']),stored['top1_joint'])
            check_reloaded(model,stored,samples,state,'cuda:0')
            p0=decode_original(model,state['p0_simcc'],samples)
            original=model.add_pred_to_datasample(model.head.predict(feats,samples,model.test_cfg),None,deepcopy(samples))
            delta=0.
            for prefix,results in [('P_0',p0),('original',original)]:
                coords=np.stack([array(r.pred_instances.keypoints)[0] for r in results])
                scores=np.stack([array(r.pred_instances.keypoint_scores)[0] for r in results])
                np.testing.assert_allclose(coords,stored[prefix+'_coords'],atol=1e-4,rtol=1e-6)
                np.testing.assert_allclose(scores,stored[prefix+'_scores'],atol=1e-6,rtol=1e-6)
                delta=max(delta,float(np.max(np.abs(coords-stored[prefix+'_coords']))))
            report.append(dict(shard=path.name,count=len(ds),max_coordinate_abs_difference=delta))
    atomic_json(a.output,dict(passed=True,batches=report,checked_count=sum(r['count'] for r in report),
                             checks=['live features','uncertainty','top1','P0/original coords and scores','alpha0','mixed candidate reload']))
    print('PASS',report)


if __name__=='__main__':main()
