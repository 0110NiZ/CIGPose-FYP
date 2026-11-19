# Module B：全量训练与 val2017 AP 选模

> 当前主实验：全量 train2017 训练，按 val2017 AP 选模。最佳第 1 轮，AP=73.6726，详见 [新版结果](results_ap/README.md)。这是最佳验证结果，不是独立测试成绩。


本版对齐 A/C 已说明的数据使用和选模方式：使用全部合格 train2017 人体更新 B，
不再额外留出内部开发集；每轮进行完整 val2017 检测框验证，按官方 COCO AP 保存最佳轮次。
同分保留较早轮次。结果属于“选出的最佳验证结果”，不是独立测试成绩。

## 本次改动

- 全量模式每轮训练 149,813 人，包括原先留给开发集的 14,561 人。
- 归一化统计使用全部训练人体，不使用 val2017 的特征统计或标签。
- 每轮处理 val2017 全部 104,125 个固定检测框，COCO 评分覆盖 5,000 张图片。
- 官方 CocoMetric：`bbox_keypoint`、关键点分数阈值 0.2、OKS NMS 阈值 0.9、单视图。
- 选择指标改为未四舍五入的 `coco/AP`；同分保留较早轮次。
- 训练损失、B 网络、输入特征、四种 α、学习率、随机种子和默认 30 轮保持原 B 设置。
- CIGPose 继续冻结。val2017 只参与模型选择，不反向传播，也不进入归一化统计。
- 旧全量 gain 标签直接复用，不重新生成。

本次对齐的是训练数据使用和最佳模型选择协议，未声称不同模块的网络、损失和超参数完全相同。
若组内另有统一训练轮数预算，可在新实验开始时通过 `--epochs` 设置。
新结果是否更好，需要实际训练验证。

## 保留旧实验

原实验目录和已经提交的 `Module_B/results`、`history.zip` 都保留。
覆盖更新包中的代码文件即可，不要删除整个 Module_B 文件夹，不要修改旧报告的哈希和路径。

| 用途 | 新目录 |
| --- | --- |
| 1 轮流程检查 | `/mnt/e/fyp_module_b/train_b_ap_smoke` |
| 正式 30 轮训练 | `/mnt/e/fyp_module_b/train_b_ap` |
| 最佳模型四组对照评估 | `/mnt/e/fyp_module_b/eval_b_ap` |
| 新报告与最佳权重的 GitHub 快照 | `Module_B/results_ap` |

`model.py`、`common.py`、`labels.py` 不需要修改。新增 `validate.py` 负责每轮完整 AP 验证。
旧模型仍可通过历史源代码哈希检查进行评估，但不会被误标为采用本版 AP 选模协议。

## 先做 1 轮流程检查

在 WSL 的 cigpose 环境、项目根目录运行：

```bash
python Module_B/train.py --repo /mnt/f/FYP/CIGPose-FYP-git --cache /mnt/e/CIGPose_shared_cache_full_20261112/train2017 --labels /mnt/e/fyp_module_b/full_train_v4 --val-cache /mnt/e/val_detection_cache/val2017-detection --ann /mnt/f/FYP/data/coco/annotations/person_keypoints_val2017.json --checkpoint /mnt/f/FYP/CIGPose-FYP-git/checkpoints/cigpose-m_coco_256x192.pth --output /mnt/e/fyp_module_b/train_b_ap_smoke --smoke --epochs 1
```

小检查只减少训练人数到约 4,096；这些人全部参与训练，不再内部划分。
验证仍是完整 104,125 个检测框。结束时会重新加载最佳权重，再做一次完整验证复核十项指标。
因此 1 轮小检查也会出现两次 val 进度，比之前的内部开发集检查慢得多。
这份小模型的 AP 只检查流程，不用于最终比较，也不会被正式完整模型评估入口接受。

## 正式训练

小检查通过后，使用独立新目录重新初始化 B：

```bash
python Module_B/train.py --repo /mnt/f/FYP/CIGPose-FYP-git --cache /mnt/e/CIGPose_shared_cache_full_20261112/train2017 --labels /mnt/e/fyp_module_b/full_train_v4 --val-cache /mnt/e/val_detection_cache/val2017-detection --ann /mnt/f/FYP/data/coco/annotations/person_keypoints_val2017.json --checkpoint /mnt/f/FYP/CIGPose-FYP-git/checkpoints/cigpose-m_coco_256x192.pth --output /mnt/e/fyp_module_b/train_b_ap --full
```

每轮应看到 `trained 149813 people`，然后是完整 val2017 进度。
`history.json` 记录每轮 loss、官方十项指标和所选强度分布。
`best.pt` 按 AP 选择，不按 loss、OKS gain 或最后一轮选择。
最终再次读取最佳模型做完整验证，十项指标必须复现。
`best_predictions.keypoints.json` 保存其后处理结果。

## 中断后续跑

保持代码、环境、输入和所有参数不变，在原命令末尾加 `--resume`。
每轮归档在 `epochs/epoch_XXX.pt`；`last.pt` 是已完成轮次的提交点，包含优化器、历史和随机状态。
如果中断发生在某轮训练或验证中，该轮会从上一完整提交点重做。
脚本根据 last.pt 修复 best.pt 和 history.json，避免半次保存造成错位。
初始第 0 轮也有检查点，因此第一次验证中断也可续跑。

正常 Ctrl+C 会清理写锁。若 WSL 被强制关闭而遗留锁，可在同一恢复命令加
`--resume --recover-lock`。它只允许恢复同一主机上已不存在的 PID，不允许两个训练进程同时写入。
不允许拿旧开发集实验的目录续跑本版，也不允许在续跑时改变轮数或代码。

## 训练完成后的四组对照

固定最佳权重，比较 P0、固定 α=0.25、Top-1 α=1 和 B：

```bash
python Module_B/evaluate.py --repo /mnt/f/FYP/CIGPose-FYP-git --cache /mnt/e/val_detection_cache/val2017-detection --ann /mnt/f/FYP/data/coco/annotations/person_keypoints_val2017.json --checkpoint /mnt/f/FYP/CIGPose-FYP-git/checkpoints/cigpose-m_coco_256x192.pth --b-run /mnt/e/fyp_module_b/train_b_ap --output /mnt/e/fyp_module_b/eval_b_ap --full
```

新完整训练的 AP 必须与此处 B 的 AP 一致。
此步骤继续核对官方、公共 CPU 入口、最终 JSON 的十项指标，以及后处理记录。
val2017 已经参与选模，报告会明确标记它不是独立测试。

## 收集新报告

完整四组评估结束后：

```bash
python Module_B/result.py
```

默认读 train_b_ap、eval_b_ap 和既有 full_train_v4，写入新目录 `Module_B/results_ap`。
逐字节复制报告和 best.pt；同时保存本次代码的 source.zip。
旧的 `Module_B/results` 不覆盖。结论与 AP 差值根据新结果动态生成，不预设 B 一定提升。
完整 gain、特征缓存、逐轮权重和大型预测 JSON 继续保留在 E 盘，不复制进 Git。

## 验证状态

交付前完成了命令入口、全量使用模式、AP 选模与并列处理、完整验证覆盖、来源检查、
历史检查点恢复和报告归档的本地测试。
评分桥接使用模拟组件测试了数据格式与完整性；交付环境没有 PyTorch/GPU，
实际 CUDA 训练、原生 CocoMetric 和恢复训练须在用户的 cigpose 环境验证。

本地协议测试可在项目根目录运行：

```bash
python -m unittest discover -s Module_B/tests -p 'test_*.py'
```

这些测试包含模拟评分组件，用于检查接口、覆盖率和来源记录，不替代真实 GPU 验证。
