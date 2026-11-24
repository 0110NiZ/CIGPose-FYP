"""Persist epoch diagnostics, training metrics, and the KD controller state."""
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from mmengine.hooks import Hook

from c_tracking import build_loss_loader, evaluate_loss, preserve_rng
from gradients import (BATCH_COUNT, build_batches, choose_factor, contribution,
                       probe, summarize_gram)

STATE_KEY = 'group_e_adaptive'
ADAPTIVE_EPOCHS = 80
BATCH_MEANS = ('loss_kpt', 'loss_cf', 'kd_raw', 'loss_kd',
               'feature_raw', 'loss_feature', 'loss')
GATE_SUMS = (
    'gate_eligible', 'gate_passed',
    'gate_passed_teacher_error_sum', 'gate_passed_student_error_sum',
    'gate_filtered_teacher_error_sum', 'gate_filtered_student_error_sum',
)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_batches(runner, work, manifest, stage_one, resume):
    path = work / 'diagnostic_batches.pth'
    if resume:
        if sha256_file(path) != manifest['diagnostic_sha256']:
            raise RuntimeError('Diagnostic batch cache differs from the original run.')
        return torch.load(path, map_location='cpu', weights_only=False)

    batches = build_batches(runner, stage_one)
    if len(batches) != BATCH_COUNT:
        raise RuntimeError('Expected exactly five diagnostic batches.')
    temp = path.with_suffix('.tmp')
    torch.save(batches, temp)
    temp.replace(path)
    manifest['diagnostic_sha256'] = sha256_file(path)
    manifest['diagnostic_batches'] = [
        dict(batch=i, sample_ids=[str(s.id) for s in batch['data_samples']],
             image_ids=[str(s.img_id) for s in batch['data_samples']],
             flip=[bool(s.metainfo.get('flip', False)) for s in batch['data_samples']])
        for i, batch in enumerate(batches, 1)
    ]
    return batches


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AdaptiveTracking(Hook):
    priority = 'LOW'

    def __init__(self, work, manifest, stage_one, resume):
        self.work = work
        self.manifest = manifest
        self.stage_one = stage_one
        self.resume = resume
        self.restored = None
        self.rows = []
        self.ap = []
        self.steps = 0
        self.factor = 1.
        self.anneal_start = None

    def after_load_checkpoint(self, runner, checkpoint):
        if self.resume:
            self.restored = checkpoint.get(STATE_KEY)

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint[STATE_KEY] = dict(
            experiment=self.manifest['recipe'], diagnostic_sha256=self.manifest['diagnostic_sha256'],
            base_weights=self.base, control=self.control, anneal_start=self.anneal_start,
            loss_history=self.rows, ap_history=self.ap,
        )

    def before_train(self, runner):
        self.base = {name: runner.model.train_cfg[name] for name in ('kd_weight', 'feature_weight')}
        manifest_path = self.work / 'run_manifest.json'
        if self.resume:
            saved = json.loads(manifest_path.read_text())
            state = self.restored
            if (state is None or state['control']['epoch'] != runner.epoch
                    or state['experiment'] != self.manifest['recipe']
                    or saved['recipe'] != self.manifest['recipe']
                    or state['diagnostic_sha256'] != saved['diagnostic_sha256']
                    or state['base_weights'] != self.base):
                raise RuntimeError('Checkpoint, source, or training recipe differs from the original adaptive run.')
            self.manifest = saved
            self.control = state['control']
            self.factor = self.control['common_factor']
            self.anneal_start = state['anneal_start']
            runner.model.train_cfg.update(self.control['weights_after'])
            self.rows = list(state['loss_history'])
            self.ap = list(state['ap_history'])
            ap_path = self.work / 'ap_history.json'
            if ap_path.is_file():
                # Validation may have completed after the regular checkpoint was written.
                by_epoch = {row['epoch']: row for row in self.ap}
                by_epoch.update({row['epoch']: row for row in json.loads(ap_path.read_text())
                                 if row['epoch'] <= runner.epoch})
                self.ap = [by_epoch[epoch] for epoch in sorted(by_epoch)]
            self.save_losses()
            write_json(ap_path, self.ap)
            for path in (self.work / 'gradient_epochs').glob('epoch_*.json'):
                if int(path.stem.split('_')[-1]) > runner.epoch:
                    path.unlink()
        else:
            self.manifest['student_initialized_sha256'] = state_digest(runner.model)
        self.batches = cache_batches(runner, self.work, self.manifest, self.stage_one, self.resume)
        write_json(manifest_path, self.manifest)
        self.loader = build_loss_loader(runner)

        def stepped(optimizer, args, kwargs):
            self.steps += 1

        runner.optim_wrapper.optimizer.register_step_post_hook(stepped)
        if (self.resume and runner.epoch % runner.train_loop.val_interval == 0
                and not any(row['epoch'] == runner.epoch for row in self.ap)):
            with preserve_rng():
                runner.val_loop.run()

    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        old_factor = self.factor
        probe_started = time.perf_counter()
        try:
            records = probe(runner, self.batches, self.base)
        except torch.cuda.OutOfMemoryError:
            # Restore state inside probe, then keep the previous valid factor.
            records = []
        before = contribution(records, old_factor)
        if epoch <= ADAPTIVE_EPOCHS:
            phase = 'adaptive'
            multiplier = 1.
            self.factor, search = choose_factor(records, old_factor, before)
        else:
            phase = 'annealing'
            if self.anneal_start is None:
                self.anneal_start = old_factor
            multiplier = (runner.max_epochs - epoch) / (runner.max_epochs - ADAPTIVE_EPOCHS)
            self.factor = self.anneal_start * multiplier
            search = dict(status='annealing', candidates=[], hit_limit=False)

        after = contribution(records, self.factor)
        self.control = dict(
            epoch=epoch, phase=phase, anneal_multiplier=multiplier,
            diagnostic_seconds=time.perf_counter() - probe_started,
            common_factor_before=old_factor, common_factor=self.factor,
            relative_factor=self.factor / old_factor if old_factor else None,
            projection_before=before, projection_after=after,
            target_reached=(0.40 <= after <= 0.50) if after is not None and phase == 'adaptive' else None,
            weights_before={key: base * old_factor for key, base in self.base.items()},
            weights_after={key: base * self.factor for key, base in self.base.items()},
            search=search,
        )
        batches = []
        for record in records:
            grams = record.get('grams', {})
            batches.append(dict(batch=record['batch'], loss_scale=record['loss_scale'],
                                groups={scope: dict(before=summarize_gram(gram, old_factor),
                                                    after=summarize_gram(gram, self.factor),
                                                    base_gram=gram.tolist())
                                        for scope, gram in grams.items()}))
        path = self.work / 'gradient_epochs'
        path.mkdir(exist_ok=True)
        write_json(path / f'epoch_{epoch:03d}.json',
                   dict(control=self.control, batches=batches))
        runner.model.train_cfg.update(self.control['weights_after'])
        if before is None:
            runner.logger.warning(f'Incomplete gradient probe at epoch {epoch}')
        runner.logger.info('KD_CONTROL ' + json.dumps({key: value for key, value in self.control.items()
                                                        if key != 'search'} |
                                                       {'search_status': search['status']}))
        self.sums = dict.fromkeys(BATCH_MEANS + GATE_SUMS, 0.)
        self.people = 0
        self.iterations = 0
        self.first_steps = self.steps
        self.lr_start = self.lrs(runner)
        self.train_start = time.perf_counter()

    @staticmethod
    def lrs(runner):
        reasoning = {id(param) for module in (runner.model.head.cim, runner.model.head.gcn)
                     for param in module.parameters()}
        groups = runner.optim_wrapper.optimizer.param_groups
        return {label: sorted({group['lr'] for group in groups
                               if (id(group['params'][0]) in reasoning) == is_reasoning})
                for label, is_reasoning in (('cim_gnn', True), ('other', False))}

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        count = len(data_batch['data_samples'])
        names = BATCH_MEANS + GATE_SUMS
        values = torch.stack([outputs[name].detach().float() for name in names]).tolist()
        for name, value in zip(names, values, strict=True):
            if not math.isfinite(value):
                raise FloatingPointError(f'Non-finite {name}')
            self.sums[name] += value * count if name in BATCH_MEANS else value
        self.people += count
        self.iterations += 1

    def save_losses(self):
        if self.rows:
            temp = self.work / 'loss_history.csv.tmp'
            with temp.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            temp.replace(self.work / 'loss_history.csv')

    def after_train_epoch(self, runner):
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - self.train_start
        assert self.people == 44941 and self.iterations == 352
        val = evaluate_loss(runner.model, self.loader)
        assert val['val_people'] == 6352
        row = dict(epoch=runner.epoch + 1, train_people=self.people,
                   train_seconds=train_seconds, **val,
                   train_loss_kpt_per_person=self.sums['loss_kpt'] / self.people,
                   optimizer_steps=self.steps - self.first_steps,
                   skipped_optimizer_steps=self.iterations - (self.steps - self.first_steps),
                   lr_start=self.lr_start, lr_end=self.lrs(runner),
                   phase=self.control['phase'])
        row.update({f'train_{name}_batch_mean': self.sums[name] / self.people for name in BATCH_MEANS})
        eligible, passed = (int(self.sums[key]) for key in ('gate_eligible', 'gate_passed'))
        filtered = eligible - passed
        row.update(gate_eligible=eligible, gate_passed=passed, gate_filtered=filtered,
                   gate_pass_rate=passed / eligible if eligible else None)
        for label, count in (('passed', passed), ('filtered', filtered)):
            for role in ('teacher', 'student'):
                key = f'gate_{label}_{role}_error'
                row[f'{key}_mean_input_px'] = self.sums[f'{key}_sum'] / count if count else None
        self.rows.append(row)
        self.save_losses()
        runner.logger.info('LOSS_AND_GATE ' + json.dumps(row))

    def after_val_epoch(self, runner, metrics=None):
        row = dict(epoch=runner.epoch, **{key: float(value) for key, value in metrics.items()})
        self.ap = [record for record in self.ap if record['epoch'] != runner.epoch] + [row]
        write_json(self.work / 'ap_history.json', self.ap)
