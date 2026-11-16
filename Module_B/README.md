# Module B：干预强度选择

B 在 CIGPose 选出关节后，选择干预强度 `0.25 / 0.5 / 0.75 / 1.0`。
本轮固定使用不确定性 Top-1 选关节，冻结 CIGPose，只训练 B。
当前工作是基于缓存的 B-only 训练与评估，尚未合入 CIGPose 的正式实时推理入口。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `config.py` | 强度、特征格式和默认路径 |
| `common.py` | 文件校验、报告保存 |
| `data.py` | 特征输入、标签对齐、图片划分 |
| `model.py` | B 的小型 MLP 网络 |
| `labels.py` | 生成四种强度的真实 gain |
| `train.py` | 训练 B |
| `evaluate.py` | 检测框 val2017 评估 |
| `result.py` | 收集本轮报告和最佳权重，准备上传 GitHub |
| `history.zip` | 已完成实验使用的原始代码，仅用于核对报告中的哈希 |

网络仍为 `563 → 128 → 64 → 4`，输入、归一化、损失函数和强度选择规则保持原实现。
`model.py` 是训练和评估共用的同一个模型定义。
原实验代码的字节没有改动，放在 `history.zip` 中；平时只使用本文件夹中的新入口。
协议格式里的 `v1`、`v4` 等是数据格式标记，保留以识别现有实验，不是多个运行脚本。
仓库已有的 `mmpose/analysis/fyp` 公共接口名称保持不变。

## 先收集已完成的结果

把整个 `Module_B` 文件夹放到仓库根目录：

```text
F:\FYP\CIGPose-FYP-git\Module_B
```

在 WSL 的 `cigpose` 环境执行：

```bash
cd /mnt/f/FYP/CIGPose-FYP-git
python Module_B/result.py
```

默认从以下现有目录读取，不移动或修改其中的文件：

- `/mnt/e/fyp_module_b/full_train_v4`
- `/mnt/e/fyp_module_b/train_b_full_v1`
- `/mnt/e/fyp_module_b/eval_b_full_v1`

脚本先核对标签、训练、权重和评估报告的对应关系，再逐字节复制到 `Module_B/results`。
收集内容包括标签报告、训练划分与曲线、最佳 B 权重、四组评估报告。
完整特征缓存、全量 gain 分片、原始预测 JSON 和基础 CIGPose 大权重不复制进 Git。
报告中原来的绝对路径与哈希不改写，因为这些描述的是已经完成的历史实验。
文件夹内的 Git 属性保留结果文件的原始字节，避免 Windows 换行转换影响校验。
`result.py` 用于归档本轮原始实验；未来重新实验可另行指定新的归档目录与记录方式。

## 检查整理后的评估入口

整理后的入口允许读取原始完整训练的 `best.pt`。它要求原训练脚本哈希与存档一致，
同时继续校验报告、权重、特征格式和网络结构；不会伪装成用新文件名重新训练过。

如已执行 `result.py`，可以用复制进仓库的权重检查两个分片：

```bash
python Module_B/evaluate.py --repo /mnt/f/FYP/CIGPose-FYP-git --cache /mnt/e/val_detection_cache/val2017-detection --ann /mnt/f/FYP/data/coco/annotations/person_keypoints_val2017.json --checkpoint /mnt/f/FYP/CIGPose-FYP-git/checkpoints/cigpose-m_coco_256x192.pth --b-run /mnt/f/FYP/CIGPose-FYP-git/Module_B/results/train --output /mnt/e/fyp_module_b/eval_b_clean_smoke --smoke-shards 2
```

这是新的小规模推理检查，不会重新训练，也不覆盖已完成的正式评估。
正式比较仍保留原四组评估报告。新入口尚需在目标 GPU 环境通过此检查。

## 已完成的正式结果

COCO val2017，固定检测框，单视图；104,125 个检测实例，评估覆盖 5,000 张图片。

| 方法 | AP |
| --- | ---: |
| P0，不干预 | 73.6223 |
| 固定 α=0.25 | **73.6652** |
| Top-1，α=1 | 73.5402 |
| B，自适应强度 | 73.6263 |

四组官方 CocoMetric、公共 CPU 入口和最终 JSON 的十项指标均一致，后处理记录也一致。
B 优于 Top-1，但未超过固定 α=0.25；相对 P0 的差异很小，未做统计显著性声明。
本次 B 最佳轮数为 3。模型使用全部 train2017 合格人体中的 135,252 人训练，
14,561 人作为按图片隔离的内部开发集。GT 不进入 B 的输入。

## 以后重新运行

这些是重构后的新入口，写入**新输出目录**，不要用它们续跑旧脚本产生的目录。
旧实验的代码指纹、报告和已完成标签均保留。现有标签不必重算。

训练小样本：

```bash
python Module_B/train.py --repo /mnt/f/FYP/CIGPose-FYP-git --cache /mnt/e/CIGPose_shared_cache_full_20261112/train2017 --labels /mnt/e/fyp_module_b/full_train_v4 --output /mnt/e/fyp_module_b/train_b_clean_smoke --smoke
```

全量训练：

```bash
python Module_B/train.py --repo /mnt/f/FYP/CIGPose-FYP-git --cache /mnt/e/CIGPose_shared_cache_full_20261112/train2017 --labels /mnt/e/fyp_module_b/full_train_v4 --output /mnt/e/fyp_module_b/train_b_clean_full --full
```

如需重新生成标签：

```bash
python Module_B/labels.py --repo /mnt/f/FYP/CIGPose-FYP-git --cache /mnt/e/CIGPose_shared_cache_full_20261112/train2017 --ann /mnt/f/FYP/data/coco/annotations/person_keypoints_train2017.json --checkpoint /mnt/f/FYP/CIGPose-FYP-git/checkpoints/cigpose-m_coco_256x192.pth --output /mnt/e/fyp_module_b/labels_b_clean --full
```

`labels.py` 保留最终版本的近似并列审计、源数据检查、确定性检查和保存回读校验。
一次性的旧 v3 分片迁移代码已从当前入口移除；已经迁移完成的标签及其来源记录不变。
训练注释文件的哈希须与完整原始训练缓存的协议一致。
同一新入口、代码和参数未改变时，可用 `--resume` 续跑它自己创建的目录。

完整评估时，将上述评估命令的 `--smoke-shards 2` 改为 `--full`，并使用新的输出目录。

## 上传

先收集结果并完成小规模推理检查，再暂存 `Module_B` 文件夹。
本文件夹之外的旧脚本和缓存先保留；不要用 `git add .` 把无关临时文件一起提交。
历史 `smoke_b_labels.py` 已在原仓库提交 `f758e60` 中，当前新入口不再依赖该脚本。
