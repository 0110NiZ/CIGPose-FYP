import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from Module_B import common, config, validate, result
from test_protocol import row


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.lp,self.tp,self.ep=[self.root/x for x in ('labels','train','evaluate')]
        for p in (self.lp,self.tp,self.ep):p.mkdir()
        self.history=self.root/'history.zip'
        with zipfile.ZipFile(self.history,'w') as z:z.writestr('generate_b_labels_v4.py',b'original label source')
        lr=dict(scope='full',code_sha256={'/repo/scripts/generate_b_labels_v4.py':hashlib.sha256(b'original label source').hexdigest()})
        common.atomic_json(self.lp/'run.json',lr)
        common.atomic_json(self.lp/'summary.json',dict(scope='full',passed=True,fingerprint=common.digest(lr),instances=149813,candidate_records=599252))
        (self.lp/'resolved_config.py').write_text('flip_test=False\n')
        split=dict(protocol=validate.PROTOCOL,train_annotation_ids=list(range(149813)),
            train_image_ids=[100000],development_annotation_ids=[],development_image_ids=[],validation_image_ids=list(range(5000)))
        common.atomic_json(self.tp/'split.json',split)
        (self.tp/'best.pt').write_bytes(b'weights')
        self.tr=dict(schema_version='b-gain-regressor-ap-v1',protocol=validate.PROTOCOL,scope='full',
            script_sha256=common.sha(str(Path(config.__file__).with_name('train.py'))),code_sha256=common.source_hashes(*validate.TRAIN_FILES),
            feature_spec=config.FEATURE_SPEC,network=config.NETWORK,alphas=config.ALPHAS,
            train_people=149813,internal_development_people=0,epochs=3,
            val2017_used=True,val2017_role='epoch_selection_only',val2017_updates_parameters=False,
            labels_fingerprint=common.digest(lr),labels_summary_sha256=common.sha(self.lp/'summary.json'),
            split_sha256=common.digest(split),validation=dict(cache_run_sha256='cache',annotation_sha256='ann',checkpoint_sha256='checkpoint'))
        common.atomic_json(self.tp/'run.json',self.tr)
        self.hist=[row(1,.71),row(2,.72),row(3,.715)]
        common.atomic_json(self.tp/'history.json',self.hist)
        best=self.hist[1]['validation'];common.atomic_json(self.tp/'best_validation.json',best)
        ts=dict(scope='full',training_completed=True,protocol=validate.PROTOCOL,run_fingerprint=common.digest(self.tr),
            gt_in_inputs=False,train_people=149813,development_people=0,normalization_train_people=149813,
            val2017_used=True,val2017_role='epoch_selection_only',val2017_updates_parameters=False,
            independent_test=False,best_recheck_matches=True,best_epoch=2,best_validation_AP=.72,
            best_validation_metrics=best['metrics'],best_checkpoint_sha256=common.sha(self.tp/'best.pt'))
        common.atomic_json(self.tp/'summary.json',ts)
        er=dict(scope='full',b_training_fingerprint=common.digest(self.tr),b_checkpoint_sha256=common.sha(self.tp/'best.pt'),
            b_training_summary_sha256=common.sha(self.tp/'summary.json'),b_model_selection_protocol=validate.PROTOCOL,
            source_cache_run_sha256='cache',annotation_sha256='ann',checkpoint_sha256='checkpoint',
            scoring=dict(score_mode='bbox_keypoint',keypoint_score_thr=.2,nms_mode='oks_nms',nms_thr=.9))
        common.atomic_json(self.ep/'run.json',er)
        self.es=dict(scope='full',formal_coco_ap_evaluated=True,prediction_checks_passed=True,
            fingerprint=common.digest(er),detections=104125,training_selection_AP_matches=True,
            val2017_used_for_model_selection=True,independent_test=False,evaluations={},
            ap_0_to_1=dict(zip(result.MODES,[.7,.71,.69,.72])),
            b_minus_controls_ap_points={'p0_current':2.,'fixed_alpha025':1.,'top1_alpha1_current':3.})
        for mode in result.MODES:
            metrics=dict(all_10_metrics_match=True,postprocessing_records_match=True,
                         audit={'evaluation_image_count':5000},input_count=104125)
            self.es['evaluations'][mode]=metrics;(self.ep/mode).mkdir()
            common.atomic_json(self.ep/mode/'metrics.json',metrics)
        common.atomic_json(self.ep/'summary.json',self.es)
    def tearDown(self):self.temp.cleanup()
    def test_exact_copy_and_source_archive(self):
        output=self.root/'snapshot'
        result.collect(self.lp,self.tp,self.ep,output,self.history)
        self.assertEqual((output/'train/best.pt').read_bytes(),b'weights')
        with zipfile.ZipFile(output/'source.zip') as z:
            self.assertEqual(z.read('Module_B/train.py'),Path(str(Path(config.__file__).with_name('train.py'))).read_bytes())
        self.assertEqual((output/'.gitattributes').read_text(),'* -text\n')
        result.collect(self.lp,self.tp,self.ep,output,self.history)
    def test_old_result_directory_not_overwritten(self):
        output=self.root/'old_results';output.mkdir();(output/'keep.txt').write_text('old')
        with self.assertRaises(FileNotFoundError):result.collect(self.lp,self.tp,self.ep,output,self.history)
        self.assertEqual((output/'keep.txt').read_text(),'old')
    def test_wrong_comparison_AP_rejected(self):
        self.es['ap_0_to_1']['b_current']=.719
        common.atomic_json(self.ep/'summary.json',self.es)
        with self.assertRaises(ValueError):result.check_reports(self.lp,self.tp,self.ep,self.history)
    def test_wrong_val_membership_rejected(self):
        split=json.loads((self.tp/'split.json').read_text());split['development_annotation_ids']=[1]
        common.atomic_json(self.tp/'split.json',split)
        with self.assertRaises(ValueError):result.check_reports(self.lp,self.tp,self.ep,self.history)


if __name__=='__main__':unittest.main()
