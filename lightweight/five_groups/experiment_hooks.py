"""Training policy and logging for the single-GPU C-branch recipe."""
import csv
import json
from pathlib import Path

from mmengine.hooks import Hook

from c_tracking import LossTrackingHook, preserve_rng
from kd import load_teacher
from objectives import early_phase, reasoning_param_groups, transfer_blocks


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


class ExperimentPolicy(Hook):
    # Restore the scheduler's LR before the standard ParamSchedulerHook runs.
    priority = 'HIGHEST'

    def __init__(self, group, policy, teacher_path, resume=False):
        self.group = group
        self.policy = policy
        self.teacher_path = teacher_path
        self.resume = resume
        self.original_lrs = []

    def before_train(self, runner):
        student = runner.model
        with preserve_rng():
            teacher = load_teacher(next(student.parameters()).device, self.teacher_path)
        object.__setattr__(student, '_teacher', teacher)
        if not self.resume:
            transfer_blocks(student, teacher)
        self.groups = reasoning_param_groups(runner.optim_wrapper.optimizer, student)
        runner.logger.info('G%d | official CIM/GNN %s | feature=%s | initial policy=%s',
                           self.group, 'restored from run' if self.resume else 'copied and verified',
                           student.feature_mode, self.policy)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if batch_idx == 0:
            self.factor = early_phase(runner.model, self.policy, runner.epoch + 1)
        self.original_lrs = [g['lr'] for g in self.groups]
        for group, normal in zip(self.groups, self.original_lrs):
            group['lr'] = normal * self.factor
        self.normal_lr = self.original_lrs[0]
        self.reasoning_lr = self.groups[0]['lr']
        if batch_idx == 0:
            runner.logger.info('G%d epoch %d, scheduled LR %.9g, CIM/GNN LR %.9g',
                               self.group, runner.epoch + 1, self.normal_lr, self.reasoning_lr)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for group, normal in zip(self.groups, self.original_lrs):
            group['lr'] = normal
        self.original_lrs = []

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {})['five_group'] = self.group

    def after_load_checkpoint(self, runner, checkpoint):
        if self.resume and checkpoint.get('meta', {}).get('five_group') != self.group:
            raise ValueError('The resume checkpoint does not belong to this experiment group.')


class FiveGroupTracking(LossTrackingHook):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self.feature_rows = []

    def before_train(self, runner):
        super().before_train(runner)
        if self.schedule.resume:
            directory = Path(runner.work_dir)
            epoch = runner.epoch
            self.rows = [r for r in read_csv(directory / 'loss_history.csv')
                         if int(r['epoch']) <= epoch]
            self.feature_rows = [r for r in read_csv(directory / 'feature_history.csv')
                                 if int(r['epoch']) <= epoch]
            history = directory / 'ap_history.json'
            if history.is_file():
                self.ap = [r for r in json.loads(history.read_text()) if int(r['epoch']) <= epoch]
            if len(self.rows) != epoch or len(self.feature_rows) != epoch:
                raise ValueError('The resume checkpoint and epoch histories are inconsistent. '
                                 'Resume from the original experiment directory.')
            for r in self.rows:
                r['epoch'] = int(r['epoch'])

    def before_train_epoch(self, runner):
        super().before_train_epoch(runner)
        self.feature_sum = 0.
        self.weighted_sum = 0.

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        people = len(data_batch['data_samples'])
        raw = float(outputs.get('feature_raw', 0.))
        weighted = float(outputs.get('loss_feature', 0.))
        if not (float('-inf') < raw < float('inf') and float('-inf') < weighted < float('inf')):
            raise FloatingPointError('Non-finite feature loss.')
        self.feature_sum += raw * people
        self.weighted_sum += weighted * people
        self.raw_sum += weighted * people  # include the new term in C's total
        if batch_idx == 0:
            self.first_lrs = (self.schedule.normal_lr, self.schedule.reasoning_lr)

    def after_train_epoch(self, runner):
        # C hook validates 44,941 training / 6,352 validation people and writes loss_history.csv.
        super().after_train_epoch(runner)
        self.feature_rows.append(dict(
            epoch=runner.epoch + 1, group=self.schedule.group,
            raw_feature_mean=self.feature_sum / self.n,
            weighted_feature_mean=self.weighted_sum / self.n,
            lr_multiplier=self.schedule.factor, first_scheduled_lr=self.first_lrs[0],
            first_cim_gnn_lr=self.first_lrs[1], last_scheduled_lr=self.schedule.normal_lr,
            last_cim_gnn_lr=self.schedule.reasoning_lr))
        write_csv(Path(runner.work_dir) / 'feature_history.csv', self.feature_rows)
