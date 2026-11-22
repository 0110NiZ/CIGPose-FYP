# C 组基础知识蒸馏

C 组继承 [A 组配置](../configs/team128_baseline100.py)，使用同一份 30% COCO 分层训练子集、种子 21、训练 batch 128、验证 batch 64、FP16 和 100 轮调度。学生沿用 6,190,372 参数的模型，从同一份小骨干预训练权重起步。冻结的官方 CIGPose-m 教师和学生看同一张增强图，各做一次前向。

## 运行

从仓库根目录进入已配置的 Linux 训练环境，按 [A 组的数据准备说明](../README.md#环境和数据) 安装本仓库的 MMPose 并生成固定子集。下载 [官方 COCO 17 点、256×192 的 CIGPose-m 权重](https://drive.google.com/file/d/1tNXidCKVhXqxT8WIYPEdSfIEownbIl5E/view)，保存为 `checkpoints/cigpose-m-coco-256x192.pth`。脚本会核验该文件的 SHA256。

从学生初始权重开始训练 C 组：

```bash
python -u lightweight/group_c_kd/train.py train
```

默认输出为 `work_dirs/group_c_kd/`，记录每轮训练与验证关键点损失、每 10 轮的完整验证 AP 和遮挡误差，以及训练完成后的曲线。可用 `--work-dir` 更换输出位置，用 `--teacher` 指定教师权重路径。A 和 C 可以独立训练。

如需复核最初的系数标定，可以运行 `python lightweight/group_c_kd/train.py pilot`，在 `work_dirs/group_c_kd_pilot/` 查看 `calibration.json` 和教师骨架预览。正式实验直接使用已标定的系数 0.04969。

如需用同一遮挡指标评估训练好的 A 组，可以运行：

```bash
python lightweight/group_c_kd/train.py baseline \
  --baseline-checkpoint work_dirs/team128_baseline/best_coco_AP_epoch_100.pth
```

## 蒸馏和验证

学生对 CIM、GNN 后的 SimCC 横纵坐标分布做输出蒸馏。师生 logits 乘 10 后计算 FP32 的 KL（教师分布到学生分布），横纵两轴各占一半，再取有效关节均值。COCO 中有标注的遮挡点和可见点等权，编码后无效的关节跳过。总损失为 `loss_kpt + loss_cf + 0.04969 × raw_kd`。验证、保存权重和部署都使用学生模型。完整 AP 用 EMA 学生和翻转测试；遮挡误差按原图坐标计算全部 8,365 个 `v=1` 关节的平均欧氏距离。

| 第 100 轮验证 | AP（0–100） | 遮挡误差（原图像素） |
| --- | ---: | ---: |
| A 无蒸馏基线 | 70.34 | 17.633 |
| C 基础蒸馏 | 70.43 | 17.945 |

在仓库根目录运行 `python lightweight/group_c_kd/eval_teacher_occlusion.py`，可复测官方教师开启与关闭翻转测试的 AP 和遮挡误差。A、C 均训练到 100 轮后，运行 `python lightweight/group_c_kd/audit_pose_metrics.py` 可在 CPU 上用同一批标注人体框统计教师、A、C 的可见及遮挡关节误差。使用 PyTorch 2.8 环境时可附加 `--flops`，统计 256×192 输入的单人单次前向。默认读取上面两个训练目录中的第 100 轮 checkpoint；自定义目录使用 `--a-work-dir` 和 `--c-work-dir`。
