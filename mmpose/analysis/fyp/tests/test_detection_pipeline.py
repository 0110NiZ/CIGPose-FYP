"""Protocol regressions: cache identity/resume, small-area OKS NMS and scores."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
FYP = Path(__file__).resolve().parents[1]
ROOT = FYP.parents[2]
sys.path[:0] = [str(FYP), str(ROOT/'scripts'), str(ROOT/'mmpose')]
from detection_cache import ReferenceCache, CacheWriter, SHAPES, INTEGER_FIELDS
from eval_unified import UnifiedPoseEvaluator, _oks_nms, COCO_SIGMAS


def arrays(indices=(0,1)):
    n=len(indices)
    a={k:np.zeros((n,)+shape,dtype=np.int64 if k in INTEGER_FIELDS else np.float32)
       for k,shape in SHAPES.items()}
    a['source_detection_index'][:]=indices
    a['image_id'][:]=42
    a['confidence'][:]=1
    a['bbox'][:]=[0,0,10,20]
    for k in ('input_scale','input_size','original_size'):a[k][:]=20
    a['bbox_score'][:]=.7
    return a


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'cache'
        self.protocol=dict(dataset_split='val2017',source_detection_sha256='a'*64,box_protocol='detection')
        self.cache=ReferenceCache.create(self.path,self.protocol,[(42,0),(42,1)])
    def tearDown(self):self.tmp.cleanup()
    def test_resume_partial_and_complete(self):
        with CacheWriter(self.cache) as w:w.append(arrays((0,)))
        with self.assertRaises(ValueError):self.cache.verify(True)
        resumed=ReferenceCache.create(self.path,self.protocol,[(42,0),(42,1)],resume=True)
        with CacheWriter(resumed) as w:w.append(arrays((1,)))
        self.assertEqual(resumed.verify(True),{0,1})
        self.assertEqual(resumed.instance_id(1),'val2017:det:'+'a'*64+':000001')
    def test_resume_changed_detector_rejected(self):
        with self.assertRaises(ValueError):
            ReferenceCache.create(self.path,dict(self.protocol,source_detection_sha256='b'*64),[(42,0),(42,1)],True)
    def test_duplicate_rejected(self):
        with CacheWriter(self.cache) as w:
            w.append(arrays())
            with self.assertRaises(ValueError):w.append(arrays())
    def test_wrong_image_rejected(self):
        a=arrays();a['image_id'][0]=99
        with CacheWriter(self.cache) as w:
            with self.assertRaises(ValueError):w.append(a)
    def test_gt_field_rejected(self):
        a=arrays();a['annotation_id']=np.array([1,2])
        with CacheWriter(self.cache) as w:
            with self.assertRaises(ValueError):w.append(a)
    def test_lock_and_roundtrip(self):
        with CacheWriter(self.cache) as w:
            with self.assertRaises(FileExistsError):
                with CacheWriter(self.cache):pass
            a=arrays();w.append(a)
        loaded=self.cache.load_shard(self.cache.shards()[0])
        for k in a:np.testing.assert_array_equal(a[k],loaded[k])


class EvaluatorTests(unittest.TestCase):
    def test_nms_matches_reference_different_and_small_areas(self):
        from mmpose.evaluation.functional import oks_nms
        for scale in (1.,.0001):
            coords=np.zeros((3,17,2),np.float32)
            coords[1,:,0]=4.5*np.sqrt(scale);coords[2,:,0]=100*np.sqrt(scale)
            areas=np.array([10000,40000,20000],np.float32)*scale
            scores=np.array([.9,.8,.7])
            db=[dict(keypoints=np.column_stack([p,np.ones(17,np.float32)]),area=areas[i],score=scores[i]) for i,p in enumerate(coords)]
            self.assertEqual(_oks_nms(coords,scores,areas),oks_nms(db,.9,sigmas=COCO_SIGMAS).tolist())
    def test_missing_scales_and_cross_batch_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'ann.json'
            f.write_text(json.dumps(dict(images=[dict(id=42,width=100,height=100)],annotations=[],categories=[dict(id=1,name='person')],info={})))
            ev=UnifiedPoseEvaluator(f)
            r=dict(image_id=42,detection_instance_id='d1',keypoints=np.zeros((17,2)).tolist(),
                   keypoint_scores=np.ones(17).tolist(),bbox=[0,0,10,20],bbox_score=.5)
            with self.assertRaises(ValueError):ev.add_records([r])
            ev.reset();r['input_scale']=[12.5,25];ev.add_records([r])
            with self.assertRaises(ValueError):ev.add_records([r])


if __name__=='__main__':unittest.main()
