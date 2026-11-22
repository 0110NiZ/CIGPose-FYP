import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / "lightweight/configs/team128_baseline100.py"
sys.path[:0] = [str(HERE), str(ROOT / "mmpose")]
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536 if hard == resource.RLIM_INFINITY else min(65536, hard), hard))

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from kd import DEFAULT_TEACHER, load_teacher
from c_tracking import LossTrackingHook, preserve_rng
from mmengine.hooks import Hook


def teacher_preview(inputs, targets, samples, work):
    from PIL import Image, ImageDraw
    edges = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
             (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    xy = torch.stack([z.argmax(-1) for z in targets], -1).cpu().numpy()/2
    candidates = []
    for i, sample in enumerate(samples):
        labels = sample.gt_instance_labels
        gt = torch.stack([z.argmax(-1) for z in (labels.keypoint_x_labels,labels.keypoint_y_labels)],-1)[0].cpu().numpy()/2
        valid = labels.keypoint_weights[0].cpu().numpy()>0
        if valid.sum() >= 8:
            candidates.append((i,gt,valid))
    sheet = Image.new("RGB",(192*4,280*2))
    for j,(i,gt,valid) in enumerate(candidates[:8]):
        rgb=inputs[i].permute(1,2,0).float().cpu().numpy()*[58.395,57.12,57.375]+[123.675,116.28,103.53]
        im=Image.fromarray(np.uint8(rgb.clip(0,255)))
        draw=ImageDraw.Draw(im)
        for a,b in edges:
            draw.line([tuple(xy[i,a]),tuple(xy[i,b])],fill="cyan",width=2)
        for k in np.flatnonzero(valid):
            x,y=gt[k];draw.ellipse((x-2,y-2,x+2,y+2),fill="yellow")
        x,y=xy[i,9];draw.text((x,y),"LW",fill="white")
        off=(j%4*192,j//4*280)
        sheet.paste(im,off)
        ImageDraw.Draw(sheet).text((off[0]+3,off[1]+256),"teacher cyan | GT yellow",fill="white")
    assert candidates
    sheet.save(work/"teacher_preview.jpg",quality=90)


def pilot(runner):
    runner._init_model_weights()
    model=runner.model.train()
    d=next(iter(runner.train_dataloader))
    data=model.data_preprocessor(d,training=True)
    imgs,samples=data["inputs"],data["data_samples"]
    with torch.no_grad():
        targets=model._teacher.head(model._teacher.extract_feat(imgs))
    assert [tuple(z.shape) for z in targets]==[(128,17,384),(128,17,512)]
    teacher_preview(imgs,targets,samples,Path(runner.work_dir))
    with torch.autocast("cuda",dtype=torch.float16):
        losses=model(**data,mode="loss")
    kpt,cf,kd=[float(losses[k].detach()) for k in ("loss_kpt","loss_cf","kd_raw")]
    assert np.isfinite([kpt,cf,kd]).all() and kd>0
    weight=round(0.10*kpt/kd,5)
    grad=torch.autograd.grad(losses["loss_kd"],model.head.cls_gx.weight)[0]
    assert grad.isfinite().all() and grad.abs().max()>0
    assert all(p.grad is None for p in model._teacher.parameters())
    report=dict(temperature_equivalent=0.1,lambda_kd=weight,kpt=kpt,cf=cf,
                kd_raw=kd,weighted_kd=weight*kd,people=len(samples),
                student_params=sum(p.numel() for p in model.parameters()))
    assert report["student_params"]==6190372
    (Path(runner.work_dir)/"calibration.json").write_text(json.dumps(report,indent=2))
    print(report,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["pilot","train","baseline"])
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path,
                        default=ROOT/"work_dirs/team128_baseline/best_coco_AP_epoch_100.pth")
    args = parser.parse_args()
    mode = args.mode
    defaults = {"train": "group_c_kd", "pilot": "group_c_kd_pilot", "baseline": "group_a_eval"}
    work = (args.work_dir or ROOT/"work_dirs"/defaults[mode]).resolve()
    ck = args.teacher.resolve()
    baseline_checkpoint = args.baseline_checkpoint.resolve()
    os.chdir(ROOT/"mmpose")
    cfg = Config.fromfile(str(BASELINE if mode == "baseline" else HERE/"config.py"))
    cfg.work_dir = str(work)
    if mode=="baseline":
        cfg.load_from=str(baseline_checkpoint)
        cfg.test_evaluator=[cfg.test_evaluator, dict(type="OccludedJointError")]
        metrics=Runner.from_cfg(cfg).test()
        (work/"baseline_metrics.json").write_text(json.dumps(metrics,indent=2))
        return
    if not ck.is_file():
        raise FileNotFoundError(f"Place the official CIGPose-m checkpoint at {ck} or use --teacher")
    cfg.val_evaluator=[cfg.val_evaluator,dict(type="OccludedJointError")]
    if mode=="pilot":
        cfg.model.train_cfg.kd_weight = 1.
    else:
        work.parent.mkdir(parents=True, exist_ok=True)
        lock=(work.parent/(work.name+".lock")).open("a")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if Path(cfg.work_dir).exists(): raise RuntimeError("C results already exist")
    runner=Runner.from_cfg(cfg)
    with preserve_rng():
        teacher=load_teacher(next(runner.model.parameters()).device, ck)
    if mode=="pilot":
        object.__setattr__(runner.model,"_teacher",teacher)
        pilot(runner)
        return
    class AttachTeacher(Hook):
        def before_train(self, runner):
            object.__setattr__(runner.model, "_teacher", teacher)
    runner.register_hook(AttachTeacher())
    runner.register_hook(LossTrackingHook())
    start=time.perf_counter()
    runner.train()
    secs=time.perf_counter()-start
    hist=json.loads((Path(cfg.work_dir)/"ap_history.json").read_text())
    summary=dict(final=hist[-1],best=max(hist,key=lambda r:r["coco/AP"]),training_seconds=secs)
    (work/"summary.json").write_text(json.dumps(summary,indent=2))
    from plot_c import plot
    plot(cfg.work_dir)


if __name__=="__main__":
    main()
