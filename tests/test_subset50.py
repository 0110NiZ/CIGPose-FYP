"""CPU checks; no MMPose/CUDA required. Run from the unpacked package root."""
import ast
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'lightweight/group_e_adaptive'
sys.path.insert(0, str(CODE))
import subset50 as mod


class Dataset:
    def __init__(self, path, rows):
        self.ann_file = str(path)
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def get_data_info(self, i):
        return dict(id=self.rows[i]['id'], raw_ann_info=self.rows[i])


class DefaultSampler:
    world_size = 1
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


class Loader:
    batch_size, drop_last = 128, False
    def __init__(self, path, rows):
        self.dataset = Dataset(path, rows)
        self.sampler = DefaultSampler(len(rows))
    def __len__(self): return (len(self.dataset) + 127) // 128


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'ann.json'
        self.rows = [dict(id=i, image_id=i, keypoints=[1, 2, 2] * 17) for i in range(129)]
        self.annotations = {row['id']: row for row in self.rows}
        self.identity = dict(version='test', path=str(self.path), sha256='test', images=129, annotation_instances=129)
        self.loader = Loader(self.path, self.rows)
        self.guard = patch.object(mod, 'inspect_annotations', return_value=(self.identity, self.annotations))
    def tearDown(self): self.tmp.cleanup()
    def test_tail_batch_and_steps(self):
        with self.guard:
            report, ids, _ = mod.inspect_loader(self.loader, self.path)
        self.assertEqual((report['dataset_people'], report['batches_per_epoch'], report['planned_train_iterations']), (129, 2, 200))
        self.assertEqual(len(ids), 129)
    def test_wrong_loaded_file(self):
        self.loader.dataset.ann_file = 'wrong.json'
        with self.guard, self.assertRaises(ValueError): mod.inspect_loader(self.loader, self.path)
    def test_duplicate_loaded_ids(self):
        self.loader.dataset.rows = [self.rows[0]] * 129
        with self.guard, self.assertRaises(ValueError): mod.inspect_loader(self.loader, self.path)
    def test_drop_last_rejected(self):
        self.loader.drop_last = True
        with self.guard, self.assertRaises(ValueError): mod.inspect_loader(self.loader, self.path)
    def test_wrong_batch_size(self):
        self.loader.batch_size = 64
        with self.guard, self.assertRaises(ValueError): mod.inspect_loader(self.loader, self.path)
    def test_replacement_rejected(self):
        self.loader.sampler.replacement = True
        with self.guard, self.assertRaises(ValueError): mod.inspect_loader(self.loader, self.path)
    def test_wrong_gt_rejected(self):
        self.loader.dataset.rows = copy.deepcopy(self.rows)
        self.loader.dataset.rows[0]['keypoints'][0] = 99
        with self.guard, self.assertRaises(ValueError): mod.inspect_loader(self.loader, self.path)
    def test_foreign_annotation_rejected(self):
        self.loader.dataset.rows = copy.deepcopy(self.rows)
        self.loader.dataset.rows[0]['id'] = 999
        with self.guard, self.assertRaises(ValueError): mod.inspect_loader(self.loader, self.path)
    def test_fixed_file_rejects_other_json(self):
        self.path.write_text('{}')
        with self.assertRaises(ValueError): mod.inspect_annotations(self.path)
    def test_resume_confined_to_own_directory(self):
        source = ast.parse((CODE / 'train.py').read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'resume_checkpoint')
        namespace = dict(re=re)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'resume_checkpoint', 'exec'), namespace)
        work = self.root / 'run'
        work.mkdir()
        good = work / 'epoch_10.pth'
        good.touch()
        marker = work / 'last_checkpoint'
        marker.write_text(str(good))
        self.assertEqual(namespace['resume_checkpoint'](work), (good, 10))
        outside = self.root / 'epoch_20.pth'
        outside.touch()
        marker.write_text(str(outside))
        with self.assertRaises(FileNotFoundError): namespace['resume_checkpoint'](work)
    def test_actual_steps_and_checkpoint_counters(self):
        source = ast.parse((CODE / 'tracking.py').read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'AdaptiveTracking')
        ns = dict(Hook=object, time=time, json=json, STATE_KEY='group_e_adaptive',
                  BATCH_MEANS=(), torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
                  evaluate_loss=lambda *args: dict(val_people=6352))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'tracking', 'exec'), ns)
        hook = ns['AdaptiveTracking'](self.root, dict(recipe={}, diagnostic_sha256='cache'), [], False)
        hook.people, hook.iterations = 129, 2
        hook.expected_people, hook.expected_iterations = 129, 2
        hook.steps, hook.first_steps, hook.total_iterations = 7, 6, 10
        hook.train_start, hook.loader, hook.lr_start = time.perf_counter(), None, {}
        hook.control, hook.base = dict(phase='adaptive'), {}
        hook.sums = dict(loss_kpt=1, gate_eligible=0, gate_passed=0)
        for label in ('passed', 'filtered'):
            for role in ('teacher', 'student'):
                hook.sums[f'gate_{label}_{role}_error_sum'] = 0
        hook.lrs = lambda runner: {}
        hook.save_losses = lambda: None
        runner = SimpleNamespace(model=None, epoch=1, logger=SimpleNamespace(info=lambda msg: None))
        hook.after_train_epoch(runner)
        row = hook.rows[-1]
        self.assertEqual((row['train_iterations'], row['optimizer_steps'], row['skipped_optimizer_steps']), (2, 1, 1))
        self.assertEqual((row['cumulative_train_iterations'], row['cumulative_optimizer_steps']), (12, 7))
        checkpoint = {}
        hook.before_save_checkpoint(runner, checkpoint)
        self.assertEqual(checkpoint['group_e_adaptive']['cumulative_optimizer_steps'], 7)
        self.assertEqual(checkpoint['group_e_adaptive']['cumulative_train_iterations'], 12)


class CacheChecks(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'diagnostic_batches.pth'
        self.destination = self.root / 'copied.pth'
        self.annotation = dict(id=7, image_id=9, keypoints=[1, 2, 2] * 17)
        sample = SimpleNamespace(id=7, img_id=9, metainfo=dict(raw_ann_info=self.annotation),
                                 gt_instances=dict(transformed_keypoints=True))
        self.batches = [dict(inputs=torch.ones(128, 1), data_samples=[sample] * 128) for _ in range(5)]
        torch.save(self.batches, self.source)
        recipe = {k: 1 for k in ('seed', 'temperature', 'simcc_base_weight', 'feature_base_weight',
                  'base_lr', 'batch_size', 'epochs', 'diagnostic_augmentation', 'student_backbone',
                  'student_cim_gnn_initialization')}
        recipe.update(teacher=dict(sha256='teacher'), data=dict(val_annotations=dict(sha256='val')),
                      sources={'lightweight/group_e_adaptive/model.py': 'model'})
        self.saved = dict(recipe=recipe, diagnostic_sha256=mod.digest(self.source),
                          diagnostic_batches=[dict(sample_ids=['7'] * 128) for _ in range(5)])
        self.manifest = copy.deepcopy(self.saved)
        self.write_manifest()
    def write_manifest(self):
        (self.root / 'run_manifest.json').write_text(json.dumps(self.saved))
    def tearDown(self): self.tmp.cleanup()
    def run_reuse(self, loaded=None, annotations=None):
        return mod.reuse_diagnostics(self.source, self.destination, self.manifest,
                                     {7} if loaded is None else loaded,
                                     {7: self.annotation} if annotations is None else annotations)
    def test_verbatim_copy(self):
        batches, report = self.run_reuse()
        self.assertEqual(report['status'], 'reused_verbatim')
        self.assertEqual(self.source.read_bytes(), self.destination.read_bytes())
        self.assertTrue(self.torch.equal(batches[0]['inputs'], self.batches[0]['inputs']))
    def test_missing_member_falls_back(self):
        batches, report = self.run_reuse(loaded=set())
        self.assertIsNone(batches)
        self.assertEqual(report['status'], 'generated')
        self.assertFalse(self.destination.exists())
    def test_corrupted_cache(self):
        with self.source.open('ab') as f: f.write(b'changed')
        with self.assertRaises(ValueError): self.run_reuse()
    def test_recipe_mismatch(self):
        self.manifest['recipe']['temperature'] = .2
        with self.assertRaises(ValueError): self.run_reuse()
    def test_annotation_mismatch(self):
        with self.assertRaises(ValueError): self.run_reuse(annotations={7: dict(id=7)})


if __name__ == '__main__':
    unittest.main()
