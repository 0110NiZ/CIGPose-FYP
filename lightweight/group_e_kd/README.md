# E 组特征蒸馏与 CIM、GNN 微调

E 组从与 C 组相同的小骨干预训练权重起步。学生初始化完成后，训练脚本将官方 CIGPose-m 的 `head.cim` 和 `head.gcn` 权重及缓冲区完整复制给学生一次。学生的关节编码层与坐标预测层沿用 C 组的初始化。教师在训练期间固定，部署时只使用 6,190,372 参数的学生。

教师和学生各看一次同一张增强图，沿用 C 组最终 SimCC 分布的 KL 蒸馏。在两个模型各自的 GAU 输出、进入 CIM 的位置，取每个关节的 256 维特征，逐关节计算 `1 − cosine(student, teacher)`，再对增强后有效的关节取均值。可见点和遮挡点等权。特征损失用 FP32 计算。

```text
loss = loss_kpt + loss_cf + 0.04969 × SimCC_KL + 0.1 × feature_distance
```

`config.py` 中的 `feature_weight=0.1` 是新增损失的初始实验系数。独立的 `lr_factor=0.1` 仅用于学生 CIM 和 GNN 的学习率。按日志从 1 开始计轮，第 1–10 轮这两部分使用当步常规学习率的 0.1 倍，第 11 轮第一步起使用 1 倍。两部分从第一步起就可更新。其他学生参数全程跟随 C 组原有的预热和余弦调度。

在仓库根目录使用与 C 组相同的 Linux 训练环境、[固定子集和官方教师权重](../group_c_kd/README.md)，运行：

```bash
python -u lightweight/group_e_kd/train.py
```

默认输出目录为 `work_dirs/group_e_kd/`，指定其他位置可加 `--work-dir`。教师权重的默认路径为 `checkpoints/cigpose-m-coco-256x192.pth`，也可用 `--teacher` 指定。训练沿用固定 30% 子集、batch 128、FP16 和 100 轮。每轮记录训练和验证关键点损失，训练记录另含加权特征损失。每 10 轮评估完整验证集 AP 和遮挡关节误差。

## 已完成实验

本目录是已训练完成的余弦特征蒸馏 E 组。第 100 轮 AP 为 70.5190、AP50 为 90.5493、AP75 为 77.4368，8,365 个遮挡关节的平均误差为 17.122293 原图像素，最高 AP 轮次为 100。

- [全部指标、实际运行配置及曲线](results/README.md)
- [初始状态与第 100 轮梯度诊断及适用范围](diagnostics/README.md)
