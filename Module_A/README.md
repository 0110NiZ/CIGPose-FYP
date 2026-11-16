# Module_A

这是 WSL 中可实际运行、可修改的 A 模块目录，不是 Windows 整理包。

| 文件 | 作用 |
|---|---|
| model.py | 真正的 MLP 定义，类名 MLP，约 67,281 个参数 |
| config.py | 数据路径、输出目录、训练轮数、batch size 和学习率 |
| data.py | 复用已验收的数据读取器，按人体 ID 对齐标签 |
| train.py | 完整训练循环、定期验证、保存 best.pt/last.pt |
| evaluate.py | 单独评估已有模型，读取本目录 model.py |
| result.py | 中文显示各轮验证结果、最佳轮次和最终结果，不训练 |
| common.py | 找到项目的公共代码目录 |

先看 config.py 和 model.py，再看 train.py 的编号注释。
新 train.py 实际使用这里的 model.py，不是调用旧训练脚本的快捷方式。
数据校验、CIGPose 前向和 COCO 评估复用 ../mmpose/analysis/fyp/；不要重复修改一套。
标签已经生成并验收，生成/验收程序仍保留在公共目录，本文件夹集中放训练需要的代码。

## 运行

打开 Ubuntu-22.04 终端：

```bash
conda activate cigpose
cd /home/nzd/projects/cig-module-a-n/code/Module_A

# 只检查完整缓存和标签，不训练，也不创建训练输出目录
python train.py

# 确定要开始训练时才执行：
python train.py --run-training

# 完成训练后，单独评估 best.pt：
python evaluate.py

# 已经训练完，只查看结果，不重新训练或验证：
python result.py
```

若 conda 命令未初始化，可用 /home/nzd/miniconda3/envs/cigpose/bin/python 替代 python。

默认完整 train、20 轮、batch=256、学习率 0.001、每 5 轮完整 val。
输出在 /home/nzd/projects/cig-module-a-n/runs/module_a_mlp_seed42。
也能在命令行覆盖，例如 --epochs 20 --batch-size 256 --seed 42。
恢复同一运行加 --run-training --resume；要求配置、源码、输入标签与原运行一致。
换实验使用新的 --output 目录，不覆盖旧结果。evaluate.py 也可用 --checkpoint 和 --output 指定路径。

当前沿用旧 TF32 精度 + 全局确定性 V2。基础 CIGPose 冻结，只更新 MLP。
A-only 固定 alpha=1、始终接受；不能因为预测全负就自行回退，那属于 C。

终端输出使用中文分行显示，不再直接打印 JSON。AP/AR 显示为 0 到 100 标度，原始 history.json 和 metrics.json 仍使用 0 到 1 标度，计算方式不变。
每轮显示训练损失、收益预测误差和耗时，验证轮显示 AP/AR。训练正常结束后显示最佳轮次和最后一轮结果，并自动保存 result.txt。TXT 同时包含所有 epoch 的训练记录和各次验证指标，不只是最后一轮。
加载模型和验证时的第三方标准输出写入 model_loading.log、validation_console.log，异常仍会报告。
现有结果可以用 python result.py 查看并生成 result.txt；自定义目录用 python result.py --output 路径。
结果显示调整不重新训练，不修改现有权重、history.json 或 settings.json。源码修改后严格断点恢复检查仍会拒绝旧源码版本；查看已经完成的实验应使用 result.py，不必运行 --resume。
原公共目录中的旧 train_a.py 和 gain_predictor.py 已删除，训练和模型只保留本目录版本。
旧的 predict_a.py 仍提供公共验证函数；a_dataset.py、a_contract.py、a_runtime.py 仍为必要依赖，不能删除。
标签生成及验收入口保留。历史文档中的旧 train_a.py 命令请改用本目录的 python train.py。
如果未来修改 model.py，改变网络结构后必须使用相应的新模型权重；旧架构的 checkpoint 不保证能加载。
