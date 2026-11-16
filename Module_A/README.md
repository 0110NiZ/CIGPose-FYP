# Module_A

当前训练入口是 A2：A1 全局上下文 MLP 加关节排序损失。基础 CIGPose 冻结，A-only 固定 α=1、始终接受。

| 文件 | 作用 |
|---|---|
| model.py | A0 和 A1 全局上下文 MLP，A2 使用 a1_global 架构 |
| config.py | 本机数据路径、输出目录和 A2 默认超参数 |
| train.py | SmoothL1 加 ranking loss 的训练与逐轮验证 |
| evaluate.py | 用已有 checkpoint 验证完整 val |
| result.py | 显示已有运行结果并生成文字总结 |
| data.py、common.py | 公共数据读取和导入路径 |

数据、公共 CIGPose 前向和 COCO 评估仍复用 `mmpose/analysis/fyp/`。

## 查看结果，不训练

```bash
conda activate cigpose
cd ~/projects/cig-module-a-n/code/Module_A
python result.py --output /home/nzd/projects/cig-module-a-n/runs/module_a_a2_ranking_seed42_bs1024_eval1
```

本次 A2 的小型结果报告见 [A2 报告](../experiments/module_a/A2_ranking_seed42_bs1024/README.md)。

## 主动开始新训练

**当前 `python train.py` 会开始训练。旧版“默认只检查”和 `--run-training` 的说明不再适用。**

默认 20 轮、batch size 1024、每轮验证、学习率 0.001、seed 42、ranking 权重 0.2。
原 A2 输出目录已经有完整结果。新实验必须指定新的目录：

```bash
python train.py --output /home/nzd/projects/cig-module-a-n/runs/module_a_a2_new_run
```

`--resume` 只用于恢复配置、源码和数据一致的中断运行，不能用来继续修改后的实验。
独立验证使用 `evaluate.py --checkpoint 模型路径 --output 新的验证目录`，同样不会训练。

终端逐轮显示训练与验证信息，训练结束后显示最佳结果并保存 `result.txt`。
AP/AR 在文字总结中采用 0 到 100 标度，在 JSON 中采用 0 到 1 标度。
不能只看 loss 判断提升，需使用相同完整 val 协议比较 P₀、Top-1 和 A1。

## GitHub 上传

WSL 的仓库根目录是 `~/projects/cig-module-a-n/code`。GitHub 顶层直接显示 `Module_A/`、`experiments/`、`docs/` 等，不多套一层 `code/`。
具体命令见 [WSL 提交说明](../docs/fyp/git_usage.md)。外层 `runs/`、`labels/`、`checks/` 不在仓库内。
