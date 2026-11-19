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
| A3：全身摘要128 | model_a3_context128.py | train_a3_context128.py |
| A3：全身摘要256 | model_a3_context256.py | train_a3_context256.py |
| A3：小型Transformer | model_a3_transformer.py | train_a3_transformer.py |
| 直接MLP：带不确定性 | model_direct_with_u.py | train_direct_with_u.py |
| 直接MLP：不带不确定性 | model_direct_without_u.py | train_direct_without_u.py |
| A4：A3加风险加权Loss | model_a4.py | train_a4.py |
| A3弱选择收益Loss（0.1） | model_a3_gain_weak.py | train_a3_gain_weak.py |
| A3强选择收益Loss（0.5） | model_a3_gain_strong.py | train_a3_gain_strong.py |
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

两档选择收益实验都使用原A3网络、原SmoothL1、lr0.001、batch1024、seed42、分片缓冲打乱，训练20轮。附加项是 softmax(100×预测收益) 下的期望收益遗憾，权重分别0.1和0.5，温度固定1；推理仍为单关节argmax。训练选中收益只作监测，最佳模型只按完整验证AP选择。


## A3结构实验

摘要128/256实验只将原A3的context_dim从64改为128/256，前面的隐藏层仍为128，评分器因摘要变宽调整输入维度，其余层不变。因此256组仍经过128维中间层，不能单独证明去掉所有压缩瓶颈的效果。

小型Transformer把每个关节的同样输入（256维特征、不确定性、坐标、置信度和16维关节身份）投影为128维token，经过2层、4头、前馈256维的Transformer，再用128→64→1共享评分器预测17个收益。采用pre-norm、GELU、dropout=0；它替换A3全身压缩和打分部分，不修改冻结的CIGPose，仍argmax选一个关节、alpha=1、始终接受。

三组均从seed42重新初始化，使用原SmoothL1（收益放大100倍）、AdamW lr0.001、weight_decay0.0001、batch1024、原分片缓冲顺序，全量20轮，每轮完整验证，只保留最高AP模型。不同结构无法共享完全相同的初始权重；这里保持种子及训练条件一致。Transformer先使用相同学习率作首轮结构对比，不代表其最优设置。


## 直接输出17个收益的MLP

带不确定性版本按固定COCO-17顺序拼接每个关节的256维特征、1维不确定性、2维坐标、1维姿态置信度，得到4420维输入，使用4420→128→64→17全连接网络，前两层后为ReLU，最后一层无激活。无不确定性版本彻底去掉17个不确定性输入，为4403→128→64→17。两者都取消A3的局部分支、关节embedding和逐关节共享评分器，由输出位置对应关节身份。

保留A3输入归一化（特征LayerNorm、训练集标量统计量），坐标处理不变；无不确定性版本只为兼容公共接口接收uncertainty参数，不读取其数值。其保留的标量统计量第0项同样不参与前向计算。其余条件使用原SmoothL1、seed42、lr0.001、batch1024、weight_decay0.0001、原分片训练顺序，全量20轮且每轮完整验证；只保留最高AP模型。两组均从头训练，不是推理时屏蔽输入。CIGPose冻结、仍只选一个关节。


## A3 固定、训练 B：A+B 第一版

在 WSL 的 `cigpose` 环境、仓库根目录执行：

```bash
python Module_A/train_ab.py
```

如果已进入 `Module_A` 目录，运行 `python train_ab.py` 即可，不需要 `--run-training`。
只检查真实32人接口、标签、冻结权重和B反向传播：`python Module_A/train_ab.py --check-only`。

- `model_ab.py` 定义B网络及显式接收A选中关节的563维输入；沿用B分支f1df444的563→128→64→4、ReLU、Dropout(0.1)。不需要合并整个B分支。
- `train_ab.py` 读取统一结果文件中原A3的最佳权重，冻结A3和CIGPose；只训练B。默认seed42、30轮、batch512、AdamW lr0.001/weight_decay0.0001、收益放大1000倍后SmoothL1。模型输出为放大后的收益，选强度只取argmax。
- 首次运行自动为全149813个训练人体生成A所选关节的四个强度收益（0.25/0.5/0.75/1）。统一以原缓存P0为收益基准，不混用旧Top-1的B标签。GT仅生成监督标签，不输入模型。
- 已完成的标签分片原子保存，重用时核对权重/源码/缓存来源及分片哈希；标签生成中断可以重用。若模型或代码改变，选择新的`--labels`目录。
- 每轮完整验证104125个检测、5000张图片；执行A选关节、B选强度、一次干预、始终接受。验证最高AP保存best.pt，同分保留较早轮。
- 显示标签、训练batch、验证分片进度；每轮写入既有`experiments/module_a/results.html`。结束后重新加载最佳B权重并完整复核，再打印每轮结果、最佳轮和全部10项最终指标。
- 默认模型输出：`/home/nzd/projects/cig-module-a-n/runs/model_ab_seed42/best.pt`。该权重保存B及归一化参数，A权重路径/hash记录在settings中；部署仍需原A3权重。
- 默认标签：`/home/nzd/projects/cig-module-a-n/labels/ab_a3_selected_v1`。现有缓存/模型/标签不覆盖。
- 只保留最佳模型，因此不提供从最后一轮精确续训。训练中断后，用`--output 新目录`重新训练，可复用已生成标签。相同输出目录会拒绝覆盖。
- 可用`--epochs 20`改变预算，或`--a-checkpoint 路径`显式指定同架构A权重；换A后必须使用对应的新标签目录。数据路径沿用`config.py`。

正式全量A+B训练尚未执行，`--check-only`不代表完整AP验证已经通过。原A/B的AP不能当作A+B结果。
