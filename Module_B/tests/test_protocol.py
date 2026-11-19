import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from Module_B import config, data, common, train, evaluate, validate


def row(epoch,ap,loss=1.):
    metrics={'coco/AP':ap,**{f'm{i}':.8 for i in range(9)}}
    return dict(epoch=epoch,train_people=149813,train_scaled_huber_loss=loss,
        validation=dict(epoch=epoch,validation_AP=ap,metrics=metrics,detections=104125,evaluation_image_count=5000))


class ProtocolTests(unittest.TestCase):
    def test_ap_selection_not_last_or_lowest_loss(self):
        history=[row(1,.7,4),row(2,.72,3),row(3,.72,2),row(4,.71,1)]
        self.assertEqual(validate.check_history(history,149813)['epoch'],2)

    def test_bad_scores_rejected(self):
        for value in (float('nan'),float('inf'),-1.,1.1):
            with self.subTest(value=value),self.assertRaises(ValueError):validate.improves(value,.5)

    def test_partial_validation_rejected(self):
        for field,value in (('detections',61),('evaluation_image_count',3893)):
            history=[row(1,.7)];history[0]['validation'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate.check_history(history,149813)

    def test_missing_epoch_or_training_rows_rejected(self):
        history=[row(1,.7),row(3,.71)]
        with self.assertRaises(ValueError):validate.check_history(history,149813)
        history=[row(1,.7)];history[0]['train_people']=135252
        with self.assertRaises(ValueError):validate.check_history(history,149813)

    def test_normalization_includes_all_rows(self):
        x=np.array([[1.,2.],[3.,2.],[8.,5.]],np.float32)
        mean,std=data.normalization(x,np.ones(3,bool))
        np.testing.assert_array_equal(mean,x.mean(axis=0,dtype=np.float64).astype(np.float32))
        np.testing.assert_array_equal(std,x.std(axis=0,dtype=np.float64).astype(np.float32))

    def test_alias_recovery_uses_committed_state(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'epochs').mkdir()
            (folder/'epochs/epoch_002.pt').write_bytes(b'committed-best-2')
            (folder/'best.pt').write_bytes(b'uncommitted-best-3')
            state=dict(best_epoch=2,history=[row(1,.7),row(2,.72)])
            train.restore_aliases(folder,state)
            self.assertEqual((folder/'best.pt').read_bytes(),b'committed-best-2')
            self.assertEqual(json.loads((folder/'history.json').read_text()),state['history'])

    def test_all_data_mode_has_no_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);cache_dir=folder/'cache';label_dir=folder/'labels'
            cache_dir.mkdir();label_dir.mkdir()
            expected={i:i//2 for i in range(149813)}
            class Cache:
                directory=cache_dir
                run={'protocol':{'dataset_split':'train2017'}}
                def load_shard(self,path):
                    ids=np.array(lookup[path.name]['annotation_ids'],np.int64);n=len(ids)
                    return dict(annotation_id=ids,image_id=ids//2,top1_joint=ids%17,
                        features=np.broadcast_to(ids[:,None,None]/1000,(n,17,256)).copy().astype(np.float32),
                        confidence=np.full((n,17),.7,np.float32),uncertainty=np.full((n,17),.3,np.float32))
            Cache.expected=expected;cache=Cache()
            common.atomic_json(cache_dir/'run.json',cache.run)
            selected=data.select_ids(expected,42,2)
            infos=[]
            for start in range(0,149813,32):
                info=dict(name=f'shard-{start:012d}.npz',annotation_ids=list(range(start,min(start+32,149813))),
                    count=min(32,149813-start),source_shard_index=start//32,sha256='unused')
                if selected.intersection(info['annotation_ids']):
                    source=cache_dir/info['name'];source.write_bytes(info['name'].encode());info['sha256']=common.sha(source)
                infos.append(info)
            lookup={i['name']:i for i in infos}
            run=dict(schema_version='b-only-labels-v4',scope='full',alphas=config.ALPHAS,
                source_shards=infos,source_cache_run_sha256=common.sha(cache_dir/'run.json'))
            common.atomic_json(label_dir/'run.json',run)
            common.atomic_json(label_dir/'summary.json',dict(scope='full',passed=True,
                fingerprint=common.digest(run),instances=149813,candidate_records=599252))
            for info in infos:
                if not selected.intersection(info['annotation_ids']):continue
                a=cache.load_shard(cache_dir/info['name']);n=len(a['annotation_id'])
                before=np.full(n,.5);after=before[:,None]+np.array([[0.,.001,.002,.003]])
                payload=label_dir/info['name']
                np.savez(payload,annotation_id=a['annotation_id'],image_id=a['image_id'],joint_id=a['top1_joint'],
                    alphas=np.array(config.ALPHAS,np.float32),gain=after-before[:,None],oks_after=after,oks_before=before)
                common.atomic_json(payload.with_suffix('.json'),dict(fingerprint=common.digest(run),source=info,
                    result_sha256=common.sha(payload),source_arrays_unchanged=True,alpha1_full_replacement_features_equal=True))
            x,g,ann,images,dev,*_=data.prepare_data(cache,label_dir,42,2,split_mode='all')
            self.assertEqual(set(ann),selected)
            self.assertFalse(dev.any())
            self.assertEqual(len(x),len(selected))

    def test_aligned_and_legacy_model_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'best.pt').write_bytes(b'checkpoint')
            run=dict(schema_version='b-gain-regressor-ap-v1',protocol=validate.PROTOCOL,scope='full',
                script_sha256=common.sha(str(Path(config.__file__).with_name('train.py'))),
                code_sha256=common.source_hashes(*validate.TRAIN_FILES),network=config.NETWORK,
                feature_spec=config.FEATURE_SPEC,alphas=config.ALPHAS,train_people=149813,epochs=4,
                internal_development_people=0,val2017_used=True,val2017_role='epoch_selection_only',val2017_updates_parameters=False)
            history=[row(1,.7),row(2,.72),row(3,.72),row(4,.71)]
            summary=dict(protocol=validate.PROTOCOL,scope='full',training_completed=True,
                run_fingerprint=common.digest(run),best_checkpoint_sha256=common.sha(folder/'best.pt'),
                gt_in_inputs=False,train_people=149813,development_people=0,normalization_train_people=149813,
                val2017_used=True,val2017_role='epoch_selection_only',val2017_updates_parameters=False,
                independent_test=False,best_recheck_matches=True,best_epoch=2,best_validation_AP=.72)
            def publish(r,s,h):
                common.atomic_json(folder/'run.json',r);common.atomic_json(folder/'summary.json',s)
                common.atomic_json(folder/'history.json',h)
            publish(run,summary,history)
            evaluate.validate_b_run(folder,Path(str(Path(config.__file__).with_name('train.py'))))
            for field,value in [('best_epoch',4),('train_people',135252),('val2017_updates_parameters',True)]:
                wrong=dict(summary);wrong[field]=value;publish(run,wrong,history)
                with self.subTest(field=field),self.assertRaises(ValueError):evaluate.validate_b_run(folder,Path(str(Path(config.__file__).with_name('train.py'))))
            wrong=copy.deepcopy(history);wrong[2]['validation']['detections']=61
            publish(run,summary,wrong)
            with self.assertRaises(ValueError):evaluate.validate_b_run(folder,Path(str(Path(config.__file__).with_name('train.py'))))
            legacy=dict(schema_version='b-gain-regressor-v1',scope='full',network=config.NETWORK,
                feature_spec=config.FEATURE_SPEC,alphas=config.ALPHAS,script_sha256=config.ORIGINAL_TRAIN_SHA)
            ls=dict(scope='full',training_completed=True,gt_in_inputs=False,val2017_used=False,
                run_fingerprint=common.digest(legacy),best_checkpoint_sha256=common.sha(folder/'best.pt'))
            publish(legacy,ls,[])
            evaluate.validate_b_run(folder,Path(str(Path(config.__file__).with_name('train.py'))))


if __name__=='__main__':unittest.main()
