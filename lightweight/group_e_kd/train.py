"""Train the single-GPU Group E student using the shared COCO-30% split."""
import argparse
import fcntl
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / 'group_c_kd'), str(ROOT / 'mmpose')]
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

from mmengine.config import Config
from mmengine.runner import Runner
from c_tracking import LossTrackingHook
from kd import DEFAULT_TEACHER
from plot_c import plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, default=DEFAULT_TEACHER)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work_dirs/group_e_kd')
    args = parser.parse_args()
    teacher_path = args.teacher.resolve(strict=True)
    work = args.work_dir.resolve()
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    os.chdir(ROOT / 'mmpose')
    cfg = Config.fromfile(str(HERE / 'config.py'), lazy_import=False)
    transfer_hook, = [h for h in cfg.custom_hooks if h.type == 'TransferAndFinetuneHook']
    transfer_hook.teacher_path = str(teacher_path)
    cfg.work_dir = str(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    with (work.parent / (work.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if work.exists():
            raise FileExistsError(f'Choose a new experiment directory: {work}')
        runner = Runner.from_cfg(cfg)
        runner.register_hook(LossTrackingHook(extra_losses=('loss_feature',)))
        start = time.perf_counter()
        runner.train()
        history = json.loads((work / 'ap_history.json').read_text())
        summary = dict(final=history[-1], best=max(history, key=lambda r: r['coco/AP']),
                       training_seconds=time.perf_counter() - start)
        (work / 'summary.json').write_text(json.dumps(summary, indent=2))
        plot(work, experiment='E (SimCC + pre-CIM feature KD)')


if __name__ == '__main__':
    main()
