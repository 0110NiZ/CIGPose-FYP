"""Run E with occlusion-gated SimCC and epoch-wise gradient control."""
import argparse
import fcntl
from importlib import metadata
import os
from pathlib import Path
import re
import resource
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LIGHTWEIGHT = HERE.parent
ROOT = LIGHTWEIGHT.parent
sys.path[:0] = [str(HERE), str(LIGHTWEIGHT / 'group_e_kd'),
                str(LIGHTWEIGHT / 'group_c_kd'), str(ROOT / 'mmpose')]
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

from mmengine.config import Config
from mmengine.runner import Runner

from kd import DEFAULT_TEACHER
import model  # Register the gated head and visible-joint metric.
from plot_c import plot
from tracking import AdaptiveTracking, sha256_file, write_json


def git_revision():
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256_file(path))


def artifact_manifest(cfg, teacher):
    import json
    import torch

    train = cfg.train_dataloader.dataset
    val = cfg.val_dataloader.dataset
    paths = [
        *HERE.glob('*.py'),
        LIGHTWEIGHT / 'group_e_kd/config.py',
        LIGHTWEIGHT / 'group_e_kd/feature_kd.py',
        LIGHTWEIGHT / 'group_c_kd/config.py',
        LIGHTWEIGHT / 'group_c_kd/kd.py',
        LIGHTWEIGHT / 'group_c_kd/c_tracking.py',
        LIGHTWEIGHT / 'configs/team128_baseline100.py',
        ROOT / 'mmpose/mmpose/models/heads/coord_cls_heads/cig_head.py',
    ]
    recipe = dict(
        data=dict(
            train_annotations=file_record(Path(train.data_root) / train.ann_file),
            selected_image_ids=file_record(LIGHTWEIGHT / 'data/selected_image_ids.json'),
            subset_manifest=file_record(LIGHTWEIGHT / 'data/manifest.json'),
            val_annotations=file_record(Path(val.data_root) / val.ann_file),
            train_image_root=str((Path(train.data_root) / train.data_prefix.img).resolve()),
            val_image_root=str((Path(val.data_root) / val.data_prefix.img).resolve()),
        ),
        sources={str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)},
        seed=cfg.randomness.seed,
        teacher=file_record(teacher),
        student_backbone=cfg.model.backbone.init_cfg.checkpoint,
        student_cim_gnn_initialization='copy official teacher modules and buffers at iter 0',
        temperature=cfg.model.train_cfg.kd_temperature,
        simcc_base_weight=cfg.model.train_cfg.kd_weight,
        feature_base_weight=cfg.model.train_cfg.feature_weight,
        base_lr=cfg.optim_wrapper.optimizer.lr,
        batch_size=cfg.train_dataloader.batch_size,
        epochs=cfg.train_cfg.max_epochs,
        diagnostic_augmentation=cfg.train_dataloader.dataset.pipeline,
    )
    versions = {}
    for package in ('torch', 'mmengine', 'mmcv', 'mmdet', 'albumentations'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    # Normalize tuples so the persisted recipe can be compared after resume.
    return dict(recipe=json.loads(json.dumps(recipe)), git_commit=git_revision(),
                software=versions, device=torch.cuda.get_device_name())


def resume_checkpoint(work):
    marker = work / 'last_checkpoint'
    if not marker.is_file():
        raise FileNotFoundError(f'No last_checkpoint marker: {marker}')
    value = marker.read_text().strip()
    checkpoint = work / value
    match = re.fullmatch(r'epoch_(\d+)\.pth', checkpoint.name)
    if not value or checkpoint.resolve().parent != work or not match or not checkpoint.is_file():
        raise FileNotFoundError(f'Invalid or missing epoch checkpoint: {checkpoint}')
    return checkpoint, int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, default=DEFAULT_TEACHER)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work_dirs/group_e_adaptive')
    parser.add_argument('--temperature', type=float, choices=(0.1, 0.2), default=0.1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    teacher = args.teacher.resolve(strict=True)
    work = args.work_dir.resolve()
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    os.chdir(ROOT / 'mmpose')
    cfg = Config.fromfile(str(LIGHTWEIGHT / 'group_e_kd/config.py'), lazy_import=False)
    cfg.model.head.type = 'GatedKDHead'
    cfg.model.train_cfg.kd_temperature = args.temperature
    cfg.val_evaluator.append(dict(type='VisibleJointError'))
    cfg.train_dataloader.dataset.pipeline[-1].pack_transformed = True
    transfer, = [hook for hook in cfg.custom_hooks if hook.type == 'TransferAndFinetuneHook']
    transfer.teacher_path = str(teacher)
    switch, = [hook for hook in cfg.custom_hooks if hook.type == 'mmdet.PipelineSwitchHook']
    switch.switch_pipeline[-1].pack_transformed = True
    stage_one = cfg.train_dataloader.dataset.pipeline
    cfg.work_dir = str(work)

    work.parent.mkdir(parents=True, exist_ok=True)
    with (work.parent / (work.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and work.exists():
            raise FileExistsError(f'Choose a new work directory: {work}')
        manifest = artifact_manifest(cfg, teacher)
        if args.resume:
            checkpoint, completed_epoch = resume_checkpoint(work)
            cfg.resume, cfg.load_from = True, str(checkpoint)
            # The original hook switches at a specific epoch only.
            if completed_epoch > switch.switch_epoch:
                cfg.train_dataloader.dataset.pipeline = switch.switch_pipeline
        runner = Runner.from_cfg(cfg)
        tracking = AdaptiveTracking(work, manifest, stage_one, args.resume)
        runner.register_hook(tracking)
        runner.train()
        summary = dict(final=tracking.ap[-1],
                       best=max(tracking.ap, key=lambda row: row['coco/AP']))
        write_json(work / 'summary.json', summary)
        plot(work, experiment=f'E adaptive KD, T={args.temperature:g}')


if __name__ == '__main__':
    main()
