# A 模块第一轮训练结果

用户自行完成的全量 train2017 GT 框训练：149,813 人、seed=42、20 轮，AdamW、batch=256、学习率 0.001。基础 CIGPose 冻结，使用 V2 标签和精度。第 5/10/15/20 轮完整 val 检测框验证，按 AP 选出第 10 轮 best.pt。

## 相同 V2 协议下的结果

指标使用 0 到 100 标度，比较完整 val 的 5,000 张图和 104,125 条检测。

| 方法 | AP | AR |
|---|---:|---:|
| 原始 CIGPose | 73.5402 | 78.9704 |
| 不干预 P0 | 73.6223 | 78.9767 |
| 不确定性 Top-1，alpha=1 | 73.5402 | 78.9704 |
| A-only，MLP 选关节 | 73.5190 | 78.9814 |

- A 相对 Top-1 的 AP 变化：-0.0212 AP 点。
- A 相对 P0 的 AP 变化：-0.1033 AP 点。
- 结论：这一轮 seed 42 的 A-only **没有提高 AP**，虽然训练损失下降。不能把轻微差异解读为已证明的方法优劣；当前没有多随机种子结果。

公共基线保存在 [public-tasks 的 V2 报告](https://github.com/0110NiZ/CIGPose-FYP/tree/public-tasks/experiments/public_tasks/baselines_v2_20261116)。A 使用现有第 10 轮完整验证记录，此次没有重新训练 A，也未重新执行 A 的 GPU 推理。已核对缓存指纹、标注哈希、checkpoint 和精度 profile 一致。A 与 Top-1 都固定 alpha=1、始终接受，相同评分和 NMS；比较的主要变化是选关节策略。

## 看结果

在 WSL 的 Module_A 目录运行 `python result.py`，会打印每个 epoch 的 loss、误差和耗时、各次验证 AP/AR、最佳轮次与最后一轮，并保存 result.txt。以后训练时每轮也直接在终端显示，结束自动写 TXT。所有原始 JSON 保留 0 到 1 标度。

本目录保存 history.json、settings.json、best_metrics.json、last_metrics.json、result.txt 和 comparison.json。artifacts.json 记录权重哈希和本机路径，权重和全量预测不上传 Git。settings.json 的源码哈希对应用户训练时的版本；后续只改结果显示的代码不会回填旧来源。

本机结果目录：`/home/nzd/projects/cig-module-a-n/runs/module_a_mlp_seed42`。最佳模型是 best.pt，最后一轮是 last.pt；两者不能混用。

公共基线提交：33059ea。
