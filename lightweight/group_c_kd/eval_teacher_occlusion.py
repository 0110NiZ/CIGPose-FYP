import argparse
import json
import os
from pathlib import Path

from train import BASELINE, ROOT, Config, Runner
from kd import DEFAULT_TEACHER, TEACHER_CFG, TEACHER_SHA256, teacher_state_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate official teacher with and without flip averaging.")
    parser.add_argument('--teacher', type=Path, default=DEFAULT_TEACHER)
    parser.add_argument('--work-dir', type=Path, default=ROOT/'work_dirs/teacher_occlusion_eval')
    args = parser.parse_args()
    weights, work = args.teacher.resolve(), args.work_dir.resolve()
    os.chdir(ROOT/'mmpose')
    cfg = Config.fromfile(str(BASELINE))
    assert cfg.test_dataloader.dataset == cfg.val_dataloader.dataset
    assert cfg.test_evaluator == cfg.val_evaluator
    assert cfg.model.test_cfg.flip_test
    teacher = Config.fromfile(str(TEACHER_CFG)).model
    teacher.backbone.init_cfg = None
    teacher.test_cfg = cfg.model.test_cfg.copy()
    cfg.model = teacher
    cfg.custom_hooks = []
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = str(work)
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = [cfg.val_evaluator, dict(type='OccludedJointError')]
    runner = Runner.from_cfg(cfg)
    runner.model.load_state_dict(teacher_state_dict(weights), strict=True)
    runner.model.requires_grad_(False)
    report = dict(teacher_sha256=TEACHER_SHA256, validation_people=len(runner.test_dataloader.dataset))
    assert report['validation_people'] == 6352
    for flip, name in [(True, 'teacher_flip'), (False, 'teacher_single_view')]:
        runner.model.test_cfg['flip_test'] = flip
        metrics = runner.test()
        assert metrics['occluded/count'] == 8365
        report[name] = metrics
        (work / 'metrics.json').write_text(json.dumps(report, indent=2))
        print(name, metrics, flush=True)


if __name__ == '__main__':
    main()
