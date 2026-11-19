import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from Module_B import common, validate


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def to(self,device):return self
    def cpu(self):return self
    def numpy(self):return self.value


class Model:
    def __init__(self):self.training=True;self.bad=False
    def eval(self):self.training=False;return self
    def train(self,value=True):self.training=value;return self
    def __call__(self,x):
        result=np.tile(np.array([[.1,.3,.2,0.]],np.float32),(len(x.value),1))
        if self.bad:result[0,0]=np.nan
        return Tensor(result)


class FakeMetric:
    def __init__(self,ann_file,outfile_prefix,**kwargs):
        assert kwargs==dict(score_mode='bbox_keypoint',keypoint_score_thr=.2,nms_mode='oks_nms',nms_thr=.9)
        self.coco=SimpleNamespace(getImgIds=lambda:list(range(5000)))
        self.prefix=outfile_prefix;self.rows=[]
    def process(self,unused,records):self.rows.extend(records)
    def evaluate(self,n):
        assert len(self.rows)==n==2
        assert {r['id'] for r in self.rows}=={0,1}
        assert all('bbox_scales' in r['gt_instances'] for r in self.rows)
        Path(self.prefix+'.keypoints.json').write_text(json.dumps([{'image_id':r['img_id']} for r in self.rows]))
        return {'coco/AP':.72,**{f'm{i}':.8 for i in range(9)}}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.out=self.folder/'output';self.out.mkdir();(self.out/'epochs').mkdir()
        self.cachefolder=self.folder/'cache';self.cachefolder.mkdir()
        (self.cachefolder/'run.json').write_text('{}')
        self.ann=self.folder/'val.json';self.ann.write_text('{}')
        self.a=dict(source_detection_index=np.array([0,1]),image_id=np.array([0,0]),
            features=np.zeros((2,17,256),np.float32),top1_joint=np.array([1,2]),
            confidence=np.full((2,17),.7,np.float32),uncertainty=np.full((2,17),.3,np.float32),
            bbox=np.ones((2,4),np.float32),bbox_score=np.ones(2,np.float32),input_scale=np.ones((2,2),np.float32))
        source=self.cachefolder/'shard-000.npz';source.write_bytes(b'source')
        self.runner=validate.ValEvaluator.__new__(validate.ValEvaluator)
        self.runner.device='cpu';self.runner.ann=self.ann;self.runner.ann_sha=common.sha(self.ann)
        self.runner.run_sha=common.sha(self.cachefolder/'run.json')
        self.runner.sources=[source];self.runner.source_info=[{'sha256':common.sha(source)}]
        self.runner.pose=SimpleNamespace(head=object());self.runner.dataset_meta={}
        self.runner.images=set(range(5000))
        self.runner.cache=SimpleNamespace(directory=self.cachefolder,expected={0:0,1:0},
            load_shard=lambda p:self.a,instance_id=lambda i:'fixed:'+str(i))
        self.modules={}
        for name in ('mmpose','mmpose.evaluation','mmpose.evaluation.metrics','mmengine','mmengine.logging','build_shared_cache','torch'):
            self.modules[name]=ModuleType(name)
        self.modules['mmpose.evaluation.metrics'].CocoMetric=FakeMetric
        self.modules['mmengine.logging'].MMLogger=SimpleNamespace(get_current_instance=lambda:None)
        self.modules['torch'].inference_mode=contextlib.nullcontext
        self.modules['torch'].from_numpy=lambda x:Tensor(x)
        self.modules['torch'].as_tensor=lambda x,device=None:Tensor(x)
        self.modules['build_shared_cache'].candidate_from_cache=lambda head,a,joint,alpha,device:(
            np.broadcast_to(alpha.value[:,None,None],(2,17,2)).copy(),np.ones((2,17),np.float32))
        self.mean=np.zeros(563,np.float32);self.std=np.ones(563,np.float32)
        self.model=Model()
    def tearDown(self):self.temp.cleanup()
    def call(self,**kwargs):
        with patch.dict(sys.modules,self.modules),patch.object(validate,'EXPECTED_DETECTIONS',2):
            return self.runner.evaluate(self.model,self.mean,self.std,1000.,self.out,1,**kwargs)
    def test_official_inputs_full_image_set_and_model_mode(self):
        report,pred=self.call()
        self.assertEqual(report['validation_AP'],.72)
        self.assertEqual(report['evaluation_image_count'],5000)
        self.assertEqual(report['detections'],2)
        self.assertEqual(report['selected_alpha_counts']['0.5'],2)
        self.assertTrue(self.model.training)
        self.assertEqual(report['predictions_sha256'],common.sha(pred))
        self.assertTrue((self.out/'epochs/epoch_001.json').is_file())
    def test_recheck_has_separate_report(self):
        self.call(verify=True)
        self.assertTrue((self.out/'epochs/best_recheck.json').is_file())
        self.assertFalse((self.out/'epochs/epoch_001.json').exists())
    def test_missing_detection_rejected(self):
        self.runner.cache.expected[2]=1
        with self.assertRaises(ValueError):self.call()
        self.assertTrue(self.model.training)
    def test_changed_source_rejected(self):
        self.runner.sources[0].write_bytes(b'changed')
        with self.assertRaises(ValueError):self.call()
    def test_nonfinite_model_output_rejected(self):
        self.model.bad=True
        with self.assertRaises(ValueError):self.call()
        self.assertTrue(self.model.training)
    def test_source_mutation_rejected(self):
        original=self.modules['build_shared_cache'].candidate_from_cache
        def mutate(head,a,joint,alpha,device):
            a['features'][0,0,0]=1
            return original(head,a,joint,alpha,device)
        self.modules['build_shared_cache'].candidate_from_cache=mutate
        with self.assertRaises(AssertionError):self.call()


if __name__=='__main__':unittest.main()
