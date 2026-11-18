"""Shared complete COCO validation; temporary predictions are removed afterward."""
import argparse
import contextlib
import importlib
import json
import re
import tempfile
from pathlib import Path
import torch
import common
import config
from a_runtime import setup,load_frozen_base,require_val_cache
from detection_cache import ReferenceCache
from predict_a import validate_a

class Baseline(torch.nn.Module):
    """Top-1 or a joint chosen from full TRAIN mean gain, never from validation."""
    def __init__(self,joint=None):
        super().__init__();self.joint=joint
    def forward(self,features,uncertainty,pose_coords,pose_scores):
        if self.joint is None:return uncertainty
        scores=features.new_zeros(len(features),17);scores[:,self.joint]=1
        return scores

def evaluate_model(model,base,cache,device,provenance=None):
    with tempfile.TemporaryDirectory(prefix='module-a-val-') as temp:
        root=Path(temp)
        with (root/'console.log').open('w') as log,contextlib.redirect_stdout(log):
            record=validate_a(model.eval(),base,cache,config.VAL_ANN,root/'predictions',device,provenance)
    assert record['audit']['evaluation_image_count']==5000
    assert record['audit']['num_input_instances']==104125
    return record

def load_model(saved,device):
    if saved.get('trained') is not True or saved.get('epoch',0)<1:
        raise ValueError('Expected a trained Module A checkpoint')
    module=saved.get('model_module')
    local_module=(isinstance(module,str) and re.fullmatch(r'model_[a-z][a-z0-9_]*',module)
                  and Path(__file__).resolve().with_name(module+'.py').is_file())
    if local_module:
        model=importlib.import_module(module).MLP()
    elif saved.get('model_config',{}).get('architecture')=='a3_full_joint_context_v1':
        from model_a3 import MLP
        model=MLP(**saved['model_config'])
    elif 'model_config' in saved and saved['model_config'].get('architecture','a0') in ('a0','a1_global'):
        from model import MLP
        model=MLP(**saved['model_config'])
    else:raise ValueError('Unknown model; checkpoint must identify model_module')
    model.load_state_dict(saved['model_state'],strict=True)
    return model.to(device).eval()

if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint',type=Path);group.add_argument('--baseline',choices=['top1','fixed'])
    p.add_argument('--device',default=config.DEVICE)
    args=p.parse_args();setup(42)
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu');model=load_model(saved,args.device)
    else:
        joint=None
        if args.baseline=='fixed':
            from data import ATrainingData
            data=ATrainingData(config.TRAIN_CACHE,config.LABELS)
            joint=int(data.labels.gains.astype('float64').mean(0).argmax())
        model=Baseline(joint).eval()
    cache=ReferenceCache(config.VAL_CACHE);require_val_cache(cache);cache.verify(True)
    with contextlib.redirect_stdout(__import__('sys').stderr):base=load_frozen_base(config.BASE_MODEL,args.device)
    print(json.dumps(evaluate_model(model,base,cache,args.device)['metrics'],indent=2))
