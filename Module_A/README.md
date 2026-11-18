# Module A

模块 A 给17个候选关节预测干预收益，选择分数最高的一个。CIGPose冻结，alpha=1，始终接受。原始缓存、标签及公共完整验证协议共用。

## 实验文件

每个实验只在对应的 `model_*.py` 与 `train_*.py` 中定义差异。

新增实验时，例如在 `model_a5.py` 中提供无必填构造参数的 `MLP` 类，在 `train_a5.py` 中调用共用 `train.run('model_a5', '实验说明', ...)`。文件名使用小写英文字母、数字和下划线；评估器自动识别同目录的 `model_*.py`，不用再修改共用注册表。

| 实验 | 模型 | 训练入口 |
|---|---|---|
| A0：姿态上下文 | model_a0.py | train_a0.py |
| A1：池化全身特征 | model_a1.py | train_a1.py |
| A2：A1加排序Loss | model_a2.py | train_a2.py |
| A3：展开全身上下文 | model_a3.py | train_a3.py |
| A4：A3加风险加权Loss | model_a4.py | train_a4.py |
| 局部对照128→64 | model_local.py | train_local.py |
| 小局部64→32 | model_local_small.py | train_local_small.py |
| 局部模型，学习率0.0003 | model_local_low_lr.py | train_local_low_lr.py |

`model.py` 是历史A0/A1模型的公共实现，也供原公共代码导入，不能删除。A2/A4/低学习率实验复用对应结构，避免复制模型代码。

## 共用文件

- `train.py`：训练循环，每轮完整验证，按最高AP保存模型；相同AP取最早轮。
- `evaluate.py`：完整5000图、104125检测验证与模型加载；中间预测只临时存放。
- `data.py`、`config.py`、`common.py`：数据接口、路径配置和公共模块导入。
- `result.py`：唯一的结果文件读写和展示。

## 训练

```bash
conda activate cigpose
cd /home/nzd/projects/cig-module-a-n/code/Module_A
python train_local_small.py --run-training --output /home/nzd/projects/cig-module-a-n/runs/my_small_experiment
```

输出目录必须是新目录，防止覆盖已有实验。省略 `--run-training` 只显示实验设置。默认20轮、batch1024、seed42；学习率由实验入口指定，可用 `--lr` 覆盖。

每个实验目录只保留 `best.pt`，不保留初始、末轮或其他中间模型。因此没有“从最后一轮恢复”的功能；需要新实验时用新目录。

## 唯一结果

`/home/nzd/projects/cig-module-a-n/code/experiments/module_a/results.html`

打开即可看所有实验的每轮结果、基线和标注的最终最佳轮次。文件内同时保留完整指标数据，后续实验会更新同一个文件，不另建重复报告。

```bash
python result.py
python evaluate.py --checkpoint /home/nzd/projects/cig-module-a-n/runs/model_local_small_seed42/best.pt
python evaluate.py --baseline top1
python evaluate.py --baseline fixed
```

独立验证只显示结果，不产生新的长期结果文件。诊断脚本、诊断模型及旧的重复结果已清理。公共缓存和标签保留。
