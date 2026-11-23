"""Run from the code checkout, using its original C training environment."""
import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUP_C = HERE.parent / 'group_c_kd'


def arguments():
    parser = argparse.ArgumentParser(description='CIGPose five-group experiment')
    parser.add_argument('--group', type=int, required=True, choices=range(1, 6))
    parser.add_argument('--teacher', type=Path,
                        default=ROOT / 'checkpoints' / 'cigpose-m-coco-256x192.pth')
    parser.add_argument('--work-dir', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='one real two-person batch and backward')
    mode.add_argument('--resume', action='store_true', help='resume latest checkpoint in the same work-dir')
    return parser.parse_args()


@contextmanager
def run_directory(work, resume):
    work.parent.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(str(work) + '.lock', timeout=0):
        if resume:
            if not (work / 'last_checkpoint').is_file():
                raise FileNotFoundError(f'Missing {work / "last_checkpoint"}')
        else:
            work.mkdir(exist_ok=False)
        yield


def smoke(runner, schedule):
    import torch
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)
    if runner.param_schedulers is not None:
        runner.param_schedulers = runner.build_param_scheduler(runner.param_schedulers)
    runner._init_model_weights()
    schedule.before_train(runner)
    net = runner.model.train()
    batch = next(iter(runner.train_dataloader))
    data = net.data_preprocessor(batch, training=True)
    schedule.before_train_iter(runner, 0, batch)
    try:
        with runner.optim_wrapper.optim_context(net):
            losses = net(**data, mode='loss')
            objective = sum(v for k, v in losses.items() if k.startswith('loss'))
        objective.backward()
        if not all(torch.isfinite(v).all().item() for v in losses.values()):
            raise FloatingPointError('Non-finite smoke loss')
        upstream = net.head.final_layer.weight.grad
        if upstream is None or not torch.isfinite(upstream).all() or upstream.abs().sum() == 0:
            raise RuntimeError('Expected finite, nonzero gradient into the student encoder.')
        has_reasoning_grad = any(p.grad is not None for m in (net.head.cim, net.head.gcn)
                                 for p in m.parameters())
        assert has_reasoning_grad == (schedule.policy != 'freeze')
        assert all(p.grad is None for p in net._teacher.parameters())
        assert ('loss_feature' in losses) == (net.feature_mode != 'none')
        if net.feature_mode == 'backbone':
            grad = net.feature_projection.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        report = dict(group=schedule.group, people=len(data['data_samples']),
                      first_phase_multiplier=schedule.factor,
                      losses={k:float(v.detach()) for k,v in losses.items()})
    finally:
        schedule.after_train_iter(runner, 0)
    from objectives import early_phase
    report['epoch11_multiplier'] = early_phase(net, schedule.policy, 11)
    assert report['epoch11_multiplier'] == 1. and net.head.gcn.training
    (Path(runner.work_dir) / 'smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    args = arguments()
    teacher_path = args.teacher.resolve()
    name = f'five_group_{args.group}' + ('_smoke' if args.smoke else '')
    work = (args.work_dir or ROOT / 'work_dirs' / name).resolve()
    for p in (GROUP_C / 'kd.py', GROUP_C / 'c_tracking.py', ROOT / 'mmpose'):
        if not p.exists():
            raise FileNotFoundError(f'{p} missing. Extract this ZIP at the root of the 轻量化 branch.')
    if not teacher_path.is_file():
        raise FileNotFoundError(f'Official CIGPose-m COCO-17 checkpoint missing: {teacher_path}')
    sys.path[:0] = [str(HERE), str(GROUP_C), str(ROOT / 'mmpose')]
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    if sys.platform.startswith('linux'):
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
        if soft != resource.RLIM_INFINITY and soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    os.chdir(ROOT / 'mmpose')
    from mmengine import Config
    from mmengine.runner import Runner
    from experiment_hooks import ExperimentPolicy, FiveGroupTracking
    import five_group_model  # register FiveGroupStudent for MMEngine

    cfg = Config.fromfile(str(HERE / f'g{args.group}.py'))
    cfg.work_dir = str(work)
    cfg.resume = args.resume
    cfg.val_evaluator = [cfg.val_evaluator, dict(type='OccludedJointError')]
    if args.smoke:
        cfg.train_dataloader.update(batch_size=2, num_workers=0, persistent_workers=False)
        cfg.train_dataloader.pop('prefetch_factor', None)
    with run_directory(work, args.resume):
        runner = Runner.from_cfg(cfg)
        policy = ExperimentPolicy(args.group, cfg.early_policy, teacher_path, args.resume)
        if args.smoke:
            smoke(runner, policy)
            return
        runner.register_hook(policy)
        tracking = FiveGroupTracking(policy)
        runner.register_hook(tracking)
        runner.train()
        result = dict(group=args.group, final=tracking.ap[-1],
                      best=max(tracking.ap, key=lambda row: row['coco/AP']))
        (work / 'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
