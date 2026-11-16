"""Contract/forward tests only. No optimizer steps or real gain-label generation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

FYP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FYP))
from a_contract import (GainLabels, INPUT_KEYS, label_metadata_template, model_inputs,
                        validate_label_arrays)
from a_dataset import ATrainingData
from gain_predictor import GainPredictor, select_joint, to_device
from shared_cache import ReferenceCache, CacheWriter, hash_file, atomic_json
from train_a import gain_loss
import train_a
from a_runtime import load_a_checkpoint, EXPECTED_CHECKPOINT
from predict_a import prediction_records


def cache_arrays(ids=(10, 11, 12)):
    n = len(ids)
    rng = np.random.default_rng(12)
    u = np.tile(np.linspace(.9, .99, 17, dtype=np.float32), (n, 1))
    return dict(
        features=rng.normal(size=(n, 17, 256)).astype(np.float32), uncertainty=u,
        confidence=1-u, top1_joint=np.full(n, 16, np.int64),
        P_0_coords=rng.uniform(0, 100, (n, 17, 2)).astype(np.float32),
        P_0_scores=np.full((n, 17), .6, np.float32),
        bbox=np.tile(np.array([0, 0, 100, 200], np.float32), (n, 1)),
        bbox_score=np.ones(n, np.float32), input_center=np.full((n, 2), 50, np.float32),
        input_scale=np.full((n, 2), 100, np.float32),
        input_size=np.tile(np.array([192, 256], np.float32), (n, 1)),
        original_size=np.full((n, 2), 500, np.float32),
        image_id=np.asarray([x + 100 for x in ids], np.int64), annotation_id=np.asarray(ids, np.int64))


class ModuleATest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        protocol = dict(dataset_split='train2017', bbox_source='gt', scope='full',
                        flip_test=False, flip_protocol='single_view', base_model_eval=True,
                        checkpoint_sha256='synthetic-checkpoint', annotation_sha256='synthetic-annotation')
        self.arrays = cache_arrays()
        self.cache = ReferenceCache.create(self.root/'cache', protocol, [(110, 10), (111, 11), (112, 12)])
        with CacheWriter(self.cache) as writer:
            for sl in (slice(0, 2), slice(2, 3)):
                writer.append({k: v[sl] for k, v in self.arrays.items()})
        self.labeldir = self.root/'labels'
        self.labeldir.mkdir()
        # Synthetic labels in reverse ID order, unrelated to the source shard order.
        self.label_arrays = dict(image_id=np.array([112, 110, 111], np.int64),
                                 annotation_id=np.array([12, 10, 11], np.int64),
                                 oks_before=np.full(3, .5, np.float64),
                                 oks_after=np.array([np.linspace(.4, .6, 17),
                                                     np.linspace(.3, .7, 17),
                                                     np.linspace(.2, .8, 17)]))
        self.label_arrays['gain'] = self.label_arrays['oks_after'] - .5
        self.metadata = label_metadata_template(self.cache)
        self.metadata.update(complete=True, count=3, generator_git_commit='synthetic-fixture',
                             oks_implementation_sha256=hash_file(FYP/'oks_utility.py'))
        self.write_labels()

    def tearDown(self):
        self.tmp.cleanup()

    def write_labels(self):
        path = self.labeldir/'shard-0.npz'
        np.savez(path, **self.label_arrays)
        self.metadata['shards'] = [dict(file=path.name, sha256=hash_file(path),
                                       count=len(self.label_arrays['annotation_id']))]
        atomic_json(self.labeldir/'metadata.json', self.metadata)

    def test_id_join_ignores_file_and_row_order(self):
        labels = GainLabels(self.labeldir, self.cache)
        actual = labels.lookup(np.array([10, 11, 12]), np.array([110, 111, 112]))
        np.testing.assert_allclose(actual, self.label_arrays['gain'][[1, 2, 0]])

    def test_missing_and_wrong_image_lookup_rejected(self):
        labels = GainLabels(self.labeldir, self.cache)
        with self.assertRaises(ValueError):
            labels.lookup(np.array([999]), np.array([110]))
        with self.assertRaises(ValueError):
            labels.lookup(np.array([10]), np.array([111]))

    def test_cross_cache_labels_rejected(self):
        self.metadata['source_cache_fingerprint'] = 'wrong'
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            GainLabels(self.labeldir, self.cache)

    def test_incomplete_delivery_rejected(self):
        self.metadata['complete'] = False
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'complete'):
            GainLabels(self.labeldir, self.cache)

    def test_missing_person_rejected(self):
        self.label_arrays = {k: v[:2] for k, v in self.label_arrays.items()}
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            GainLabels(self.labeldir, self.cache)

    def test_duplicate_id_rejected(self):
        self.label_arrays['annotation_id'][1] = 12
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            GainLabels(self.labeldir, self.cache)

    def test_wrong_image_rejected(self):
        self.label_arrays['image_id'][0] = 999
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'image/person'):
            GainLabels(self.labeldir, self.cache)

    def test_gain_definition_and_nonfinite_rejected(self):
        a = {k: v.copy() for k, v in self.label_arrays.items()}
        a['gain'][0, 0] = .9
        with self.assertRaisesRegex(ValueError, 'gain !='):
            validate_label_arrays(a)
        a['gain'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_label_arrays(a)

    def test_joint_order_and_alpha_rejected(self):
        for key, value in [('joint_order', list(reversed(range(17)))), ('alpha', .5)]:
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = value
                self.write_labels()
                with self.assertRaisesRegex(ValueError, key):
                    GainLabels(self.labeldir, self.cache)
                self.metadata[key] = original

    def test_checksum_and_extra_shards_rejected(self):
        with (self.labeldir/'shard-0.npz').open('ab') as f:
            f.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            GainLabels(self.labeldir, self.cache)
        self.write_labels()
        np.savez(self.labeldir/'extra.npz', **self.label_arrays)
        with self.assertRaisesRegex(ValueError, 'manifest'):
            GainLabels(self.labeldir, self.cache)

    def test_stream_epoch_covers_all_people_including_tail(self):
        dataset = ATrainingData(self.root/'cache', self.labeldir)
        batches = list(dataset.batches(batch_size=2, buffer_shards=2, seed=7, epoch=1))
        ids = np.concatenate([b[2] for b in batches])
        self.assertEqual(sorted(ids.tolist()), [10, 11, 12])
        self.assertEqual([len(b[2]) for b in batches], [2, 1])
        again = list(dataset.batches(batch_size=2, buffer_shards=2, seed=7, epoch=1))
        np.testing.assert_array_equal(ids, np.concatenate([b[2] for b in again]))
        for x, y, bid in batches:
            self.assertEqual(set(x), set(INPUT_KEYS))
            np.testing.assert_allclose(y, dataset.labels.lookup(bid, bid+100))

    def test_input_whitelist_and_geometry_invariance(self):
        a = self.arrays
        clean = model_inputs(a)
        polluted = dict(a, gt_keypoints=np.zeros((3, 17, 3)), gain=np.ones((3, 17)),
                        gt_area=np.ones(3), annotation_id=np.full(3, 999))
        for key, value in clean.items():
            np.testing.assert_array_equal(value, model_inputs(polluted)[key])
        shifted = {k: v.copy() for k, v in a.items()}
        shifted['P_0_coords'] = shifted['P_0_coords']*2 + 30
        shifted['input_center'] = shifted['input_center']*2 + 30
        shifted['input_scale'] *= 2
        np.testing.assert_allclose(clean['pose_coords'], model_inputs(shifted)['pose_coords'], atol=2e-7)
        with self.assertRaises(ValueError):
            to_device(dict(clean, gain=np.zeros((3, 17))), 'cpu')

    def test_signed_output_and_argmax_when_all_negative(self):
        model = GainPredictor().eval()
        with torch.no_grad():
            model.regressor[-1].weight.zero_()
            model.regressor[-1].bias.fill_(-.25)
            pred = model(**to_device(model_inputs(self.arrays), 'cpu'))
        self.assertEqual(tuple(pred.shape), (3, 17))
        self.assertTrue(torch.all(pred == -.25))
        torch.testing.assert_close(select_joint(pred), torch.zeros(3, dtype=torch.long))
        pred[:, 5] = -.1
        self.assertTrue(torch.all(select_joint(pred) == 5))

    def test_forward_batch_independence_and_state_reload(self):
        model = GainPredictor().eval()
        model.set_input_statistics([.98, .6], [.001, .2])
        x = to_device(model_inputs(self.arrays), 'cpu')
        with torch.no_grad():
            pred = model(**x)
            singles = torch.cat([model(**{k: v[i:i+1] for k, v in x.items()}) for i in range(3)])
        torch.testing.assert_close(pred, singles, atol=1e-6, rtol=1e-5)
        other = GainPredictor().eval()
        other.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(pred, other(**x))

    def test_loss_gradient_without_optimizer_or_parameter_update(self):
        model = GainPredictor()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        pred = model(**to_device(model_inputs(self.arrays), 'cpu'))
        target = torch.from_numpy(self.label_arrays['gain'].astype(np.float32))
        gain_loss(pred, target).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_preflight_never_constructs_model_optimizer_or_output(self):
        # Mock only external full-data checks; exercise the actual CLI opt-in gate.
        ann = self.root/'val.json'
        atomic_json(ann, {'images': [{'id': i} for i in range(1000, 6000)]})
        output = self.root/'must-not-create'
        data = Mock(cache=self.cache, labels=Mock(fingerprint='test-labels'))
        val = Mock(expected={0: 1000}, fingerprint='test-val',
                   run={'protocol': {'source_annotation_sha256': EXPECTED_CHECKPOINT}})
        argv = ['train_a.py', '--train-cache', str(self.cache.directory),
                '--labels', str(self.labeldir), '--val-cache', 'unused',
                '--val-ann', str(ann), '--base-checkpoint', 'unused', '--output', str(output)]
        with patch.object(sys, 'argv', argv), \
                patch.object(train_a, 'ATrainingData', return_value=data), \
                patch.object(train_a, 'DetectionCache', return_value=val), \
                patch.object(train_a, 'require_cache_model'), patch.object(train_a, 'require_val_cache'), \
                patch.object(train_a, 'hash_file', return_value=EXPECTED_CHECKPOINT), \
                patch.object(train_a, 'GainPredictor', side_effect=AssertionError('model constructed')), \
                patch.object(train_a.torch.optim, 'AdamW', side_effect=AssertionError('optimizer constructed')), \
                patch.object(train_a, 'load_frozen_base', side_effect=AssertionError('base loaded')):
            train_a.main()
        self.assertFalse(output.exists())

    def test_untrained_checkpoint_cannot_enter_formal_validation(self):
        checkpoint = self.root/'untrained.pt'
        torch.save({'trained': False, 'epoch': 0}, checkpoint)
        with self.assertRaisesRegex(ValueError, 'trained Module A'):
            load_a_checkpoint(checkpoint, 'cpu')

    def test_prediction_export_preserves_detection_identity_and_score(self):
        a = dict(self.arrays, source_detection_index=np.array([8, 0, 3]))
        a['bbox_score'] = np.array([.8, .01, .6], np.float32)
        cache = Mock()
        cache.instance_id.side_effect = lambda idx: 'val:det:%d' % idx
        rows = list(prediction_records(cache, a, a['P_0_coords'], a['P_0_scores']))
        self.assertEqual([r['source_detection_index'] for r in rows], [8, 0, 3])
        self.assertEqual([r['detection_instance_id'] for r in rows], ['val:det:8', 'val:det:0', 'val:det:3'])
        self.assertEqual([r['image_id'] for r in rows], [110, 111, 112])
        np.testing.assert_allclose([r['bbox_score'] for r in rows], [.8, .01, .6])
        self.assertTrue(all('annotation_id' not in r for r in rows))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
