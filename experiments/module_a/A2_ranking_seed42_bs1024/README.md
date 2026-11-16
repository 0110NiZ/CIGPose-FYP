# A2：全局上下文 MLP 加 Ranking Loss

本实验在 A1 的全局上下文 MLP 上增加关节排序损失。基础 CIGPose 冻结，A 选择一个关节，固定 α=1，始终接受候选。

## 设置

- 完整 train2017 GT 框，149813 人；使用已验收的 A-V2 Gain 标签。
- 完整 val2017 检测框，104125 条检测，评估全部 5000 张图片；不通过 GT 删除误检。
- 20 轮，batch size 1024，每轮验证，seed 42。
- AdamW，学习率 0.001，weight decay 0.0001。
- 损失为 SmoothL1 + 0.2 × pairwise ranking；Gain 放大 100 倍，排序最小差距为放大后的 0.05。
- 沿用公共确定性 V2 设置，single-view，flip_test=False。

## 结果

指标采用 0 到 100 标度。最佳第 2 轮，AP **73.552592**，AR **78.976700**。

| 对照 | AP |
|---|---:|
| 公共缓存 P₀ | 73.622299 |
| 固定不确定性 Top-1 | 73.540217 |
| A1 全局上下文，batch 1024，每轮验证 | 73.645833 |
| A2 本次实验 | 73.552592 |

本次 A2 略高于固定 Top-1，但低于 P₀ 和 A1，不能据此认为加入 ranking loss 有效。这里是单次 seed 42 的 val 模型选择结果。

## 文件与复现

- `history.json`：每轮损失及完整验证指标，AP/AR 使用 0 到 1 标度。
- `result.txt`：原始训练输出总结，AP/AR 使用 0 到 100 标度。
- `settings.json`：训练配置、数据指纹和实际训练时的代码哈希。
- 训练代码在仓库根目录 `Module_A/`。当前默认配置为本次 A2。

已有结果只需查看上述文件。若主动开始一次新的训练，应指定新的输出目录：

```bash
conda activate cigpose
cd ~/projects/cig-module-a-n/code/Module_A
python train.py --output /home/nzd/projects/cig-module-a-n/runs/module_a_a2_new_run
```

这条命令会训练，不是检查命令。不要用它覆盖已完成的实验。
完整权重和预测留在 `/home/nzd/projects/cig-module-a-n/runs/module_a_a2_ranking_seed42_bs1024_eval1`，不纳入 Git。
