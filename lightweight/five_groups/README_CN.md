# CIGPose 轻量化 · 五组蒸馏实验

这是一份针对 `0110NiZ/CIGPose-FYP` 的增量代码包。将 ZIP 解压到已切换至 `轻量化` 分支的仓库根目录，与已有的 `lightweight`、`mmpose` 目录合并。包内文件全部位于新的 `lightweight/five_groups/`。适配并核对的提交为 `4ce9450529e9c915f11eee5b9fe2012b632e312a`。

## 训练配方

五组均从 C 组配置继承同一学生、30% 固定分层 COCO 子集、44,941 个人体实例、种子 21、256×192 输入、训练 batch 128、验证 batch 64、FP16 和 100 轮调度。使用 AIC+COCO 的 CSPNeXt-S 骨干权重。每组在学生完成基线初始化后将官方 COCO-17 教师的 `head.cim` 和 `head.gcn` 严格复制一次，包含 BatchNorm 统计量。初始坐标头、GAU 与最终分类层都按 C 组初始化。教师全程 eval、关闭梯度，教师和学生共用同一增强输入，各执行一次骨干前向。

| 组 | 新增特征损失 | 第 1–10 轮的学生 CIM/GNN | 第 11–100 轮 |
|---|---|---|---|
| 1（E 组标准） | GAU 输出，逐有效关节余弦 | 当步学习率 × 0.1，可训练 | 当步学习率 × 1 |
| 2 | 同第 1 组 | 冻结参数和 BN 运行统计 | 从第 11 轮第一步解冻，× 1 |
| 3 | 骨干输出，512→768 投影后逐位置余弦 | 当步学习率 × 0.1，可训练 | 当步学习率 × 1 |
| 4 | 同第 3 组 | 冻结参数和 BN 运行统计 | 从第 11 轮第一步解冻，× 1 |
| 5 | 仅 C 组原有坐标蒸馏 | 当步学习率 × 0.1，可训练 | 当步学习率 × 1 |

- 五组保留 `loss_kpt + loss_cf + 0.04969 × raw_kd`。横纵 SimCC 的 KL 沿用 C 组温度 0.1。
- 第 1–4 组全程加 `0.1 × raw_feature`。第 1、2 组按批内所有有效关节取平均，区分已标注和未标注点，可见点与遮挡点等权。第 3、4 组用单层无偏置 1×1 卷积，对每个人的 48 个位置沿 768 个通道计算余弦，再对所有人和位置取平均。第 3、4 组暂沿用 E 组的系数 0.1，后续可依据实测量级调参。
- 基础 AdamW 学习率约 0.0003333。C 的 1000 步线性预热及后半程余弦调度保留，仅对指定参数组的当步学习率乘倍率。冻结时保留经过 GNN 的梯度路径。
- 基础学生有 6,190,372 个参数。骨干投影额外增加 393,216 个训练参数，仅第 3、4 组在训练损失中调用。其余组没有投影层。模型验证和推理只走原有姿态预测路径。

## 使用方法

在原 C 组能运行的单 GPU 环境执行。沿用该分支现有的 MMPose/MMDetection 安装、COCO 文件布局和固定子集。官方教师权重默认放在 `checkpoints/cigpose-m-coco-256x192.pth`。使用其他路径时给命令加 `--teacher /path/to/checkpoint.pth`。加载器沿用 C 组的 SHA-256 检查。

从仓库根目录启动。若第一次设置该分支，先按 `lightweight/README.md` 准备数据并生成固定训练子集。已有 A/C 运行环境可直接执行下列命令。

```bash
# 可选，使用真实训练集的两个实例和官方教师做一次前向、反向，写入独立的 smoke 目录
python lightweight/five_groups/train.py --group 1 --smoke

# 每条都是独立的 100 轮实验
python lightweight/five_groups/train.py --group 1
python lightweight/five_groups/train.py --group 2
python lightweight/five_groups/train.py --group 3
python lightweight/five_groups/train.py --group 4
python lightweight/five_groups/train.py --group 5
```

在不同 GPU 或机器上并行时，每个进程只分配一张卡。例如在 Linux 用 `CUDA_VISIBLE_DEVICES=2 python lightweight/five_groups/train.py --group 3`。每组默认输出到独立的 `work_dirs/five_group_1` 至 `work_dirs/five_group_5`。可用 `--work-dir` 自定目录。新训练需要一个尚未使用的结果目录。断点续训示例为 `python lightweight/five_groups/train.py --group 1 --resume`，自定目录时追加原有的 `--work-dir`；续训采用已保存的学生参数、优化器和调度器状态，教师仅用于生成监督信号。

## 看结果

| 文件 | 内容 |
|---|---|
| `loss_history.csv` | 每轮训练和验证关键点损失、训练坐标蒸馏损失、优化器更新次数。训练总损失列包括新增特征项 |
| `feature_history.csv` | 每轮原始及加权特征损失、CIM/GNN 学习率倍率、首末步正常与实际学习率。第 5 组特征项为 0 |
| `ap_history.json` | 每 10 轮用完整验证集、EMA 学生及原翻转测试设置得到的 AP、遮挡关节数量和原图像素误差。AP 采用评估器的 0–1 标度，报告 AP 点数时乘 100 |
| `summary.json` | 验证 AP 最佳轮和第 100 轮的指标 |
| `smoke.json` | 运行 `--smoke` 后的两人批次检查，含分项损失和第 11 轮倍率 |

如需与 A/C 的正式结果比较，保持原来的数据、batch、EMA 和 flip test 设置。正式训练需要完整的原仓库、权重和 COCO 数据；压缩包仅含新增代码和本说明。
