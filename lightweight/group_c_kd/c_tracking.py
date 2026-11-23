"""Record exact batch aggregates without altering the baseline training objective."""
import contextlib
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.hooks import Hook


@contextlib.contextmanager
def preserve_rng():
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


def build_loss_loader(runner):
    cfg = copy.deepcopy(runner.cfg.val_dataloader)
    pipeline = cfg.dataset.pipeline
    assert pipeline[-1]['type'] == 'PackPoseInputs'
    pipeline.insert(-1, dict(type='GenerateTarget', encoder=copy.deepcopy(runner.cfg.codec)))
    with preserve_rng():
        loader = runner.build_dataloader(cfg, seed=21)
    assert len(loader.dataset) == 6352
    return loader


def evaluate_loss(model, loader, max_batches=None):
    """Raw (non-EMA) weights, eval mode, no flip, FP16, no parameter/buffer updates."""
    flags = [(m, m.training) for m in model.modules()]
    n = 0
    kpt_sum = 0.
    cf_weighted_sum = 0.
    started = time.perf_counter()
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            for i, batch in enumerate(loader):
                data = model.data_preprocessor(batch, training=False)
                count = len(data['data_samples'])
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    losses = model(**data, mode='loss')
                kpt, cf = float(losses['loss_kpt']), float(losses['loss_cf'])
                assert math.isfinite(kpt) and math.isfinite(cf)
                # Original KLDiscretLoss sums over people but divides only by K.
                # Dividing its accumulated sum by total people removes batch-size bias.
                kpt_sum += kpt
                cf_weighted_sum += cf * count
                n += count
                if max_batches is not None and i + 1 >= max_batches:
                    break
        torch.cuda.synchronize()
    finally:
        for m, training in flags:
            m.training = training
    return dict(val_people=n, val_loss_kpt_per_person=kpt_sum/n,
                val_loss_cf_batch_mean=cf_weighted_sum/n,
                val_loss_seconds=time.perf_counter()-started)


class LossTrackingHook(Hook):
    priority = 'LOW'

    def __init__(self, extra_losses=()):
        self.extra_losses = extra_losses
        self.rows = []
        self.ap = []
        self.steps = 0
        self.status = {}

    def save_status(self, runner, **kwargs):
        self.status.update(kwargs)
        p = Path(runner.work_dir)/'progress.json'
        t = p.with_suffix('.tmp')
        t.write_text(json.dumps(self.status, indent=2))
        t.replace(p)

    def before_train(self, runner):
        self.start = time.perf_counter()
        self.loader = build_loss_loader(runner)
        def stepped(opt, args, kwargs):
            self.steps += 1
        runner.optim_wrapper.optimizer.register_step_post_hook(stepped)
        self.save_status(runner, state='STARTING', epochs_planned=100,
                         batch_size=128, work_dir=runner.work_dir)

    def before_train_epoch(self, runner):
        self.n = 0
        self.kpt_sum = 0.
        self.cf_sum = 0.
        self.raw_sum = 0.
        self.kd_sum = 0.
        self.raw_kd_sum = 0.
        self.extra_sums = dict.fromkeys(self.extra_losses, 0.)
        self.batches = 0
        self.prev_steps = self.steps
        self.epoch_start = time.perf_counter()
        self.save_status(runner, state='TRAINING', epoch=runner.epoch+1, iteration=0)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        n = len(data_batch['data_samples'])
        kpt, cf = float(outputs['loss_kpt']), float(outputs['loss_cf'])
        assert math.isfinite(kpt) and math.isfinite(cf), 'Non-finite loss'
        self.n += n
        self.kpt_sum += kpt
        self.cf_sum += cf*n
        kd = float(outputs['loss_kd'].detach())
        raw_kd = float(outputs['kd_raw'].detach())
        assert math.isfinite(kd) and math.isfinite(raw_kd)
        self.kd_sum += kd*n
        self.raw_kd_sum += raw_kd*n
        self.raw_sum += (kpt+cf+kd)*n
        for name in self.extra_losses:
            value = float(outputs[name].detach())
            if not math.isfinite(value):
                raise FloatingPointError(f'Non-finite {name}')
            self.extra_sums[name] += value*n
            self.raw_sum += value*n
        self.batches += 1
        if (batch_idx+1) % 50 == 0:
            self.save_status(runner, iteration=batch_idx+1,
                             iterations=len(runner.train_dataloader),
                             train_elapsed_seconds=time.perf_counter()-self.epoch_start)

    def after_train_epoch(self, runner):
        torch.cuda.synchronize()
        train_seconds = time.perf_counter()-self.epoch_start
        assert self.n == 44941 and self.batches == 352, (self.n, self.batches)
        self.save_status(runner, state='VALIDATING_LOSS')
        result = evaluate_loss(runner.model, self.loader)
        assert result['val_people'] == 6352
        row = dict(epoch=runner.epoch+1, train_people=self.n,
                   train_loss_kpt_per_person=self.kpt_sum/self.n,
                   train_loss_cf_batch_mean=self.cf_sum/self.n,
                   train_raw_total_sample_weighted=self.raw_sum/self.n,
                   train_seconds=train_seconds,
                   train_weighted_kd_batch_mean=self.kd_sum/self.n,
                   train_raw_kd_batch_mean=self.raw_kd_sum/self.n,
                   optimizer_steps=self.steps-self.prev_steps,
                   skipped_optimizer_steps=self.batches-(self.steps-self.prev_steps),
                   lr=runner.optim_wrapper.optimizer.param_groups[0]['lr'], **result)
        row.update({f'train_{key}_batch_mean': total/self.n for key, total in self.extra_sums.items()})
        self.rows.append(row)
        with (Path(runner.work_dir)/'loss_history.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row));w.writeheader();w.writerows(self.rows)
        runner.logger.info('LOSS_CURVE '+json.dumps(row))
        self.save_status(runner, state='EPOCH_COMPLETE', last_completed_epoch=row['epoch'],
                         latest_loss=row, elapsed_seconds=time.perf_counter()-self.start)

    def after_val_epoch(self, runner, metrics=None):
        row = dict(epoch=runner.epoch, **{k:float(v) for k,v in (metrics or {}).items()})
        self.ap.append(row)
        (Path(runner.work_dir)/'ap_history.json').write_text(json.dumps(self.ap, indent=2))
        self.save_status(runner, latest_ap=row)

    def after_train(self, runner):
        assert len(self.rows) == 100 and self.rows[-1]['epoch'] == 100
        assert self.ap and self.ap[-1]['epoch'] == 100
        self.save_status(runner, state='DONE', elapsed_seconds=time.perf_counter()-self.start)
