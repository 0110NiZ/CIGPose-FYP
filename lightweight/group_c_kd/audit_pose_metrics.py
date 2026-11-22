"""Evaluate all three deployed models on identical GT crops, using CPU only."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from train import BASELINE, ROOT, Config, Runner, torch
import numpy as np
from kd import DEFAULT_TEACHER, load_teacher
from mmpose.registry import MODELS
from mmpose.utils import register_all_modules


def student(work_dir, cfg):
    model = MODELS.build(cfg.model)
    weights = work_dir / 'best_coco_AP_epoch_100.pth'
    if not weights.exists():
        weights = work_dir / 'epoch_100.pth'
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()


class SingleView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model._forward(inputs)


def compute_cost(model):
    from torch.utils.flop_counter import FlopCounterMode
    probe = torch.randn(1, 3, 256, 192)
    counter = FlopCounterMode(display=False)
    with counter:
        result = SingleView(model).eval()(probe)
    assert [x.shape[-1] for x in result] == [384, 512]
    operations = counter.get_flop_counts()['Global']
    return dict(params=sum(p.numel() for p in model.parameters()), flops=counter.get_total_flops(),
                counted_operations={str(op):count for op, count in operations.items()})


def late_training(directory):
    with (directory / 'loss_history.csv').open() as f:
        rows = {int(r['epoch']): r for r in csv.DictReader(f)}
    ap = {row['epoch']:row['coco/AP'] for row in json.loads((directory / 'ap_history.json').read_text())}
    return {epoch:dict(train=float(rows[epoch]['train_loss_kpt_per_person']),
                       val=float(rows[epoch]['val_loss_kpt_per_person']), ap=ap.get(epoch))
            for epoch in (1, 80, 90, 93, 94, 95, 100)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-work-dir', type=Path, default=ROOT/'work_dirs/team128_baseline')
    parser.add_argument('--c-work-dir', type=Path, default=ROOT/'work_dirs/group_c_kd')
    parser.add_argument('--teacher', type=Path, default=DEFAULT_TEACHER)
    parser.add_argument('--output', type=Path, default=ROOT/'work_dirs/three_model_audit.json')
    parser.add_argument('--flops', action='store_true', help='Include single-view PyTorch FLOP counts.')
    args = parser.parse_args()
    a_dir, c_dir = args.a_work_dir.resolve(), args.c_work_dir.resolve()
    teacher_weights, path = args.teacher.resolve(), args.output.resolve()
    os.chdir(ROOT/'mmpose')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    register_all_modules()
    cfg = Config.fromfile(str(BASELINE))
    cfg.model.backbone.init_cfg = None
    assert cfg.model.test_cfg.flip_test and not torch.cuda.is_available()
    models = {'teacher':load_teacher('cpu', teacher_weights),
              'baseline':student(a_dir, cfg), 'distilled':student(c_dir, cfg)}
    assert all(model.test_cfg['flip_test'] for model in models.values())
    output = {key:{} for key in models}
    config_loader = cfg.val_dataloader
    config_loader.batch_size = 32
    config_loader.num_workers = 2
    config_loader.persistent_workers = False
    loader = Runner.build_dataloader(config_loader, seed=21)
    assert len(loader.dataset) == 6352
    sums = {key:np.zeros(3, dtype=np.float64) for key in models}
    counts = np.zeros(3, dtype=np.int64)
    seen = set()
    for step, batch in enumerate(loader, 1):
        raw = [s.raw_ann_info for s in batch['data_samples']]
        gt = np.array([ann['keypoints'] for ann in raw], dtype=np.float64).reshape(-1, 17, 3)
        ids = [ann['id'] for ann in raw]
        assert len(set(ids)) == len(ids) and not seen.intersection(ids)
        seen.update(ids)
        visibility = gt[:, :, 2].astype(np.int64)
        counts += np.bincount(visibility.ravel(), minlength=3)
        for key, model in models.items():
            results = model.test_step(batch)
            assert [r.raw_ann_info['id'] for r in results] == ids
            pred = np.concatenate([r.pred_instances.keypoints for r in results])
            errors = np.linalg.norm(pred - gt[:, :, :2], axis=2)
            assert np.isfinite(errors).all()
            sums[key] += np.bincount(visibility.ravel(), weights=errors.ravel(), minlength=3)
        if step % 10 == 0 or step == 1:
            print(f'Val batches {step}/{len(loader)}', flush=True)
    assert len(seen) == 6352 and counts[1] == 8365
    for key, model in models.items():
        output[key].update(visible_px=sums[key][2]/counts[2], occluded_px=sums[key][1]/counts[1],
                           params=sum(p.numel() for p in model.parameters()))
    report = dict(validation_people=len(seen), visible_count=int(counts[2]), occluded_count=int(counts[1]),
                  models=output, baseline_history=late_training(a_dir), kd_history=late_training(c_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    if args.flops:
        for key, model in models.items():
            output[key].update(compute_cost(model))
        path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
