# 固定 50% 数据，蒸馏温度 T=0.2 对照实验

本目录是现有 T=0.1 实验代码 `lightweight/group_e_adaptive` 的独立副本。解压后保留两个目录，不要覆盖 T=0.1 代码。

训练数据仍是固定 `COCO50_nested30_fixed_annotations_seed21`，标注 SHA256 为 `bce0685bc39a2086881b4c434e931637371b68d06c2502438d271873d5a90f46`。教师、学生初始化、seed=21、batch=128、优化器、学习率、增强、验证设置及训练 100 轮均继承同一仓库配置。仅将 SimCC 蒸馏温度从 0.1 设为 0.2；不改变 KL 公式，也不手动乘温度平方或沿用 T=0.1 的蒸馏系数。

每轮训练前在固定五批增强输入上重算 GT、SimCC KD、特征 KD 与 CF 梯度。第 1–80 轮用原 `choose_factor` 将两个蒸馏权重乘以同一个新系数，目标是诊断批次上蒸馏梯度对总梯度的投影占比 40%–50%，中心 45%。若诊断无效或搜索不到区间内系数，本版本在该轮训练前报错，避免继续不符合条件的实验。该占比是五批诊断的测量值，并非所有训练 batch 的逐批保证。

第 81–100 轮沿用原退火：固定第 80 轮结束的系数，依次乘以 `(100 - epoch) / 20`。第 100 轮蒸馏系数为 0；这 20 轮不再维持 40%–50%，因为退火到 0 与持续保持该占比不能同时成立。每轮 `gradient_epochs/epoch_XXX.json` 记录实测占比、系数、阶段与权重；`run_manifest.json` 记录温度、标注哈希及配置。

## AutoDL 命令

在仓库根目录解压 T=0.2 包。沿用 T=0.1 的教师、COCO 根目录和 Python 环境：

```bash
cd /root/autodl-tmp/CIGPose-FYP-50pct
unzip /root/autodl-tmp/CIGPose_E_Adaptive_KD_50pct_T02.zip
/root/autodl-tmp/coco50-venv/bin/python -u lightweight/group_e_adaptive_t02/train.py \
  --teacher /root/autodl-tmp/checkpoints/cigpose-m-coco-256x192.pth \
  --coco-root /root/autodl-tmp/data/coco --check
```

新启动（独立目录 `work_dirs/group_e_adaptive_50pct_t02`）：

```bash
cd /root/autodl-tmp/CIGPose-FYP-50pct
/root/autodl-tmp/coco50-venv/bin/python -u lightweight/group_e_adaptive_t02/train.py \
  --teacher /root/autodl-tmp/checkpoints/cigpose-m-coco-256x192.pth \
  --coco-root /root/autodl-tmp/data/coco \
  2>&1 | tee /root/autodl-tmp/coco50_t02_train.log
```

断点续训，在同一目录用本组 checkpoint：

```bash
cd /root/autodl-tmp/CIGPose-FYP-50pct
/root/autodl-tmp/coco50-venv/bin/python -u lightweight/group_e_adaptive_t02/train.py \
  --teacher /root/autodl-tmp/checkpoints/cigpose-m-coco-256x192.pth \
  --coco-root /root/autodl-tmp/data/coco --resume \
  2>&1 | tee -a /root/autodl-tmp/coco50_t02_train.log
```

首次启动会拒绝已有 T=0.2 输出目录。不要用 T=0.1 的 checkpoint 续训 T=0.2；`--resume` 会核验本组清单和 checkpoint。五批诊断输入按相同 seed 和数据构造流程生成；若需要跨组逐字节相同的输入，应另行核验两组 `diagnostic_batches.pth` 哈希。原 `--diagnostic-cache` 来源校验要求温度相同，不能把 T=0.1 缓存直接传给 T=0.2。
