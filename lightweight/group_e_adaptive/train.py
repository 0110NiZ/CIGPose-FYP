"""Run E with occlusion-gated SimCC and epoch-wise gradient control."""
import argparse
import json
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
from subset50 import inspect_annotations, inspect_loader
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
    subset_metadata = json.loads((HERE / 'fixed_subset/manifest.json').read_text())
    if file_record(Path(val.data_root) / val.ann_file)['sha256'] != subset_metadata['source_sha256']['val']:
        raise ValueError('Validation annotations differ from the fixed subset manifest full val2017')
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
            subset_identity=inspect_annotations(train.ann_file)[0],
            subset_manifest=file_record(HERE / 'fixed_subset/manifest.json'),
            val_annotations=file_record(Path(val.data_root) / val.ann_file),
            train_image_root=str((Path(train.data_root) / train.data_prefix.img).resolve()),
            val_image_root=str((Path(val.data_root) / val.data_prefix.img).resolve()),
        ),
        sources={str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)},
        seed=cfg.randomness.seed,
        optimizer=cfg.optim_wrapper, param_scheduler=cfg.param_scheduler,
        custom_hooks=cfg.custom_hooks,
        validation=dict(dataloader=cfg.val_dataloader, evaluator=cfg.val_evaluator,
                        val_cfg=cfg.val_cfg, train_loop=cfg.train_cfg),
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
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work_dirs/group_e_adaptive_50pct_t01')
    parser.add_argument('--temperature', type=float, choices=(0.1,), default=0.1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-ann', type=Path, default=HERE / 'fixed_subset/person_keypoints_train2017_stratified50_nested30_seed21.json')
    parser.add_argument('--coco-root', type=Path, help='Optional existing COCO root; changes paths only')
    parser.add_argument('--diagnostic-cache', type=Path, help='Trusted teammate diagnostic_batches.pth, beside its run_manifest.json')
    parser.add_argument('--check', action='store_true', help='Check real dataloader without training or creating the run directory')
    args = parser.parse_args()
    ann = args.train_ann.resolve(strict=True)
    inspect_annotations(ann)
    coco_root = args.coco_root.resolve(strict=True) if args.coco_root else None
    diagnostic = args.diagnostic_cache.resolve(strict=True) if args.diagnostic_cache else None
    if args.resume and (args.check or diagnostic):
        parser.error('--resume uses this run cache; do not combine with --check/--diagnostic-cache')

    teacher = args.teacher.resolve(strict=True)
    work = args.work_dir.resolve()
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    os.chdir(ROOT / 'mmpose')
    cfg = Config.fromfile(str(LIGHTWEIGHT / 'group_e_kd/config.py'), lazy_import=False)
    cfg.train_dataloader.dataset.ann_file = str(ann)
    if coco_root:
        cfg.train_dataloader.dataset.data_root = str(coco_root)
        cfg.val_dataloader.dataset.data_root = str(coco_root)
        if cfg.get('test_dataloader'):
            cfg.test_dataloader.dataset.data_root = str(coco_root)
    if cfg.train_dataloader.batch_size != 128 or cfg.train_cfg.max_epochs != 100 or cfg.randomness.seed != 21:
        raise ValueError('Inherited config must retain batch 128, 100 epochs and seed 21')
    cfg.model.head.type = 'GatedKDHead'
    cfg.model.train_cfg.kd_temperature = args.temperature
    cfg.val_evaluator.append(dict(type='VisibleJointError'))
    # CocoMetric reads its own ann_file, independently of the val dataloader.
    # Point both at the same full val2017 JSON after resolving --coco-root.
    val_ann = (Path(cfg.val_dataloader.dataset.data_root)
               / cfg.val_dataloader.dataset.ann_file).resolve(strict=True)
    for metric in cfg.val_evaluator:
        if metric.get('ann_file') is not None:
            metric.ann_file = str(val_ann)
    if cfg.get('test_evaluator'):
        evaluators = cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]
        for metric in evaluators:
            if metric.get('ann_file') is not None:
                metric.ann_file = str(val_ann)
    cfg.train_dataloader.dataset.pipeline[-1].pack_transformed = True
    transfer, = [hook for hook in cfg.custom_hooks if hook.type == 'TransferAndFinetuneHook']
    transfer.teacher_path = str(teacher)
    switch, = [hook for hook in cfg.custom_hooks if hook.type == 'mmdet.PipelineSwitchHook']
    switch.switch_pipeline[-1].pack_transformed = True
    stage_one = cfg.train_dataloader.dataset.pipeline
    cfg.work_dir = str(work)

    if args.check:
        from mmpose.utils import register_all_modules
        from mmengine.registry import init_default_scope
        register_all_modules()
        init_default_scope(cfg.get('default_scope', 'mmpose'))
        loader = Runner.build_dataloader(cfg.train_dataloader, seed=cfg.randomness.seed)
        report, _, _ = inspect_loader(loader, ann)
        manifest = artifact_manifest(cfg, teacher)
        print(json.dumps(dict(loader=report, manifest=manifest), indent=2))
        return

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
        tracking = AdaptiveTracking(work, manifest, stage_one, args.resume, diagnostic)
        runner.register_hook(tracking)
        runner.train()
        summary = dict(final=tracking.ap[-1],
                       best=max(tracking.ap, key=lambda row: row['coco/AP']))
        write_json(work / 'summary.json', summary)
        plot(work, experiment=f'E adaptive KD 50%, T={args.temperature:g}')


if __name__ == '__main__':
    main()
