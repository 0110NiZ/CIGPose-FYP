"""Shared 128-batch runner: stop at 20 for screening without changing the schedule."""
import argparse
import os
from pathlib import Path
import sys
import resource
import fcntl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mmpose'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'lightweight/configs/team128_baseline100.py')
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--stop-after-epoch', type=int, default=100)
    p.add_argument('--resume', type=Path)
    p.add_argument('--track-val-loss', action='store_true',
                   help='Record per-person train and validation keypoint loss for a full 100-epoch run.')
    args = p.parse_args()
    if args.track_val_loss and (args.resume or args.stop_after_epoch != 100):
        p.error('--track-val-loss requires a fresh full 100-epoch run.')
    assert args.stop_after_epoch in (20, 100)
    config = args.config.resolve()
    work = args.work_dir.resolve()
    resume = args.resume.resolve() if args.resume else None
    if work.exists() and any(work.iterdir()) and resume is None:
        raise RuntimeError('Use a new work directory, or explicitly resume its checkpoint.')
    work.mkdir(parents=True, exist_ok=True)
    lock = open(work/'.training.lock', 'w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    os.chdir(ROOT/'mmpose')
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.hooks import Hook
    cfg = Config.fromfile(str(config))
    assert cfg.train_dataloader.batch_size == 128
    assert cfg.train_cfg.max_epochs == 100 and cfg.train_cfg.val_interval == 10
    assert cfg.optim_wrapper.type == 'AmpOptimWrapper'
    assert abs(cfg.optim_wrapper.optimizer.lr - 0.001/3) < 1e-12
    cfg.work_dir = str(work)
    cfg.launcher = 'none'
    cfg.resume = resume is not None
    cfg.load_from = str(resume) if resume else None
    class StopAtEpoch(Hook):
        def after_train_epoch(self, runner):
            if runner.epoch + 1 >= args.stop_after_epoch:
                runner.train_loop.stop_training = True
    runner = Runner.from_cfg(cfg)
    runner.register_hook(StopAtEpoch())
    if args.track_val_loss:
        from loss_tracking import LossTrackingHook
        runner.register_hook(LossTrackingHook())
    runner.train()
    if args.track_val_loss:
        from plot_curves import plot
        plot(work)

if __name__ == '__main__':
    main()
