# CIGPose 轻量化基线

本分支以正在使用的官方代码 `7086d7907e93fe15dce3ede7967aabe5af8c88ec` 为基线，保存当前轻量化实验；没有合入主分支中的其他创新模块。此次上传在独立副本中完成，不修改或重启 AutoDL 上正在进行的训练。

## 当前方案

| 项目 | 设置 |
| --- | --- |
| 骨干 | CSPNeXt，depth 0.33、width 0.50 |
| 参数量 | 6,190,372，原模型 14,304,732，减少 56.73% |
| Head / GCN | 保持 256 维；骨干输出接口为 512 通道 |
| 关键点 / 输入 | 17 个，宽192 × 高256 |
| CIM | 保持原版，不包含可学习干预或局部图推理改进 |
| 训练数据 | COCO train2017 固定30%分层子集：16,983张图片，44,941个有效人体 |
| 验证 | 完整 val2017，原版 GT 人体框协议；无 test2017 |
| 5090 正式训练 | FP16 AMP，batch384，100 epoch，seed21 |
| 优化 | AdamW，初始lr0.001，warmup1000 iterations；50–100轮余弦调度 |
| 后期增强 | 第93轮切换 |
| 验证 / 保存 | 每10轮，另外保留最佳 coco/AP 权重 |

分层维度为可见关键点中的遮挡比例（v=1 / v>0）、图片内人体面积中位数、有效人体数。每层按图片选约30%，保留所选图片的全部有效人体。固定种子21生成64个候选，选实例数最接近30%的候选。`data/manifest.json` 记录阈值、数量和校验摘要；`data/selected_image_ids.json` 固定最终抽样。原始抽样实现保存在 `archive/original_stratified_sampling.py`，它包含历史服务器路径，仅用于审计。

## 文件

- `configs/small30_head256_amp100.py`：可移植的完整配置，默认 AMP。
- `archive/active_5090_config.py`：当前服务器实际运行的原始配置快照，保留原路径供核对，不直接在其他机器运行。
- `scripts/prepare_subset.py`：用固定图片 ID 重建标注，并核对与实测子集完全一致。
- `scripts/train_5090.sh`：100轮启动脚本，同一 checkout 使用文件锁防止重复启动。
- `scripts/benchmark.py`：FP32/FP16单独计时；指定轮数后停止，包含完整验证。
- `results/`：5090两种精度各3轮、4060一轮的实测数据，100轮时间均为外推。
- `archive/original_5090_benchmark.py`：原始计时脚本，保留历史路径用于追溯。

## 环境与数据

使用支持 CUDA 的 Linux/WSL 环境，先激活已配置的 CIGPose 虚拟环境，再安装本仓库的 mmpose：

```bash
cd mmpose
python -m pip install -e . --no-deps
cd ..
```

5090实测环境：Python3.12、PyTorch2.8.0+cu128、CUDA12.8、MMCV2.1.0（为sm120编译）、MMEngine0.10.7、MMDetection3.2.0、NumPy1.26.4、Albumentations1.3.1。4060使用已有Python3.8、PyTorch2.0.1+cu118环境。不要直接给5090安装只支持旧架构的MMCV二进制包。其余依赖见原仓库安装文档；这里记录已验证组合，不提供未经验证的一键全新安装脚本。

将已有数据放置或链接到仓库根目录 `data/coco/`，包含 `train2017/`、`val2017/`、`annotations/`（两份官方person_keypoints JSON），并保留原版配置需要的person_detection_results文件。无需测试集。数据、权重、完整日志不上传GitHub。小骨干预训练权重由配置中的官方OpenMMLab地址首次下载。

从仓库根目录生成子集：

```bash
python lightweight/scripts/prepare_subset.py
```

生成文件 SHA256 应为 `ec3ff51924cf1be81e73e7b175f33a380b913fa51c1de5b9bd64a235d491aa5a`。该脚本只生成子集标注，不改动或删除原始数据。

## 训练与短测

在新的实验目录中启动正式训练（正在训练的服务器不需要重复执行）：

```bash
mkdir -p work_dirs
nohup bash lightweight/scripts/train_5090.sh > work_dirs/lightweight_console.log 2>&1 < /dev/null &
tail -f work_dirs/lightweight_console.log
```

运行脚本的Python必须属于已激活的CIGPose环境。checkpoint在新建的 `work_dirs/lightweight_amp100_<时间>/` 下。此锁只防止本checkout经该脚本重复启动，不能阻止其他目录或手写命令另启任务。不要与正式训练同时做GPU短测。

在仓库根目录分别运行短测：

```bash
export PYTHONPATH="$PWD/mmpose${PYTHONPATH:+:$PYTHONPATH}"
python lightweight/scripts/benchmark.py fp32 --epochs 3
python lightweight/scripts/benchmark.py amp_fp16 --epochs 3
# 笔记本4060：
python lightweight/scripts/benchmark.py amp_fp16 --epochs 1 --batch-size 64
```

短测保持100轮调度，但通过停止钩子在指定轮数后退出，每轮验证和保存。batch变化时基准学习率按batch/384调整。正式100轮训练使用标准tools/train.py，不包含短测停止钩子。短测会同步CUDA计时，因此不是完全无开销的性能测量。

## 已有实测（2026-11-19）

| 设置 | 每轮训练 | 训练整卡峰值显存 | 100轮含每10轮验证和保存预计 |
| --- | ---: | ---: | ---: |
| 5090 FP32，batch384 | 123.08秒（第2、3轮均值） | 22.08 GiB | 约3小时28分 |
| 5090 FP16 AMP，batch384 | 125.36秒（第2、3轮均值） | 15.97 GiB | 约3小时32分 |
| 4060 Laptop FP16 AMP，batch64 | 232.31秒（仅1轮） | 3.16 GiB | 约6小时38分 |

5090 AMP节省约6.1GiB整卡显存，本次短测没有显示明显速度提升。FP32沿用cuDNN允许TF32的默认设置，matmul TF32关闭；AMP为float16。4060使用不同批量、PyTorch/CUDA版本、WSL及Windows磁盘数据读取，不能将差异直接解释为纯GPU算力差。短测不是收敛或最终精度实验；当时100轮正式训练仍在进行，没有最终AP结论。

训练开始时曾误启两个同目录任务，已清理重复任务，保留一个主进程继续运行；正式训练日志首轮存在重复记录，不应将其当作独立实验。这里上传的性能数据来自此前独立完成的短测。

100轮预测会受后台负载、散热、数据缓存及第93轮增强变化影响。5090建议预算4小时；4060保持插电和当前模式建议预算7–8小时。
