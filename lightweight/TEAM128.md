# 团队统一协议与四人CIM分工

5090与4060均使用训练batch128、验证batch64。当前原版CIM基线已完成100轮，最佳AP为70.3364（第100轮）。所有候选从相同官方骨干预训练权重开始；不从已训练基线微调。

## 所有正式实验统一

配置：`lightweight/configs/team128_baseline100.py`。

| 项目 | 统一值 |
| --- | --- |
| 模型 | 小骨干depth0.33、width0.50，256维Head/GCN，17关节，输入192×256 |
| 数据 | 冻结30%图片ID；16983图片、44941有效人体；完整val2017、GT框评估 |
| 初始化 | 相同官方CSPNeXt-S预训练骨干；各CIM独立初始化，seed21 |
| batch | 单GPU128，无梯度累积；禁止自动学习率缩放 |
| 精度 | FP16 AMP，dynamic loss scale；验证FP32 |
| 优化器 | AdamW，lr=0.001/3=0.0003333333333333333，weight_decay0.05，clip_grad max_norm35 |
| 预热 | LinearLR，1000次迭代，start_factor=1e-5，约2.84轮 |
| 总调度 | 100轮；第50–100轮余弦下降，eta_min=5e-5/3；第93轮切换增强 |
| EMA/一致性损失 | 沿用统一配置；不得自行改权重或更新方式 |
| 验证/保存 | 每10轮完整验证，保留最佳coco/AP；flip_test保持配置值 |
| 数据加载 | train workers4、val workers2，batch验证64 |

学习率与最低学习率是预先约定的共同设置，不是已调优的最优超参数。正式实验请勿自行调整。

所有人使用同一份仓库提交、新建各自分支，不在共同分支直接互相覆盖。正式对比不要混用100轮从头训练与旧模型微调结果。数据生成、安装说明见README.md。不同GPU不可保证逐位相同；尽量统一PyTorch/MMCV/MMEngine/MMDetection/Albumentations版本。5090需要兼容sm120的环境。若环境无法统一，每种环境至少跑自己的原版baseline，比较同环境的AP增益；最终候选在同一环境复核。速度与显存对比必须在同一张卡、相同batch上测，不能把4060和5090速度直接归因于CIM。

## 分工：你统筹baseline，另外四人各改一个因素

当前实现位于 `mmpose/mmpose/models/heads/coord_cls_heads/cig_head.py` 的 `CausalInterventionModule`。原版选置信度最低的1个关节，以对应的可学习模板特征完全替换；训练与验证k均为1。

| 人员 / 分支建议 | 唯一改动 | 保持不变 | 验证问题 |
| --- | --- | --- | --- |
| A / cim/fixed-mix | 对选中关节使用0.5×原特征+0.5×模板 | 原置信度选1个关节 | 完全替换是否过强？ |
| B / cim/learned-strength | 对选中关节预测alpha，输出(1-alpha)×原特征+alpha×模板 | 原置信度选1个关节 | 可学习强度是否优于A的固定混合？ |
| C / cim/entropy-selection | 对x/y的softmax分布分别计算H/log(分箱数)，取均值，选熵最高1个关节 | 模板完全替换 | 更换不确定性度量是否有效？ |
| D / cim/context-replacement | 以固定骨架一跳邻居特征均值，通过小投影生成替换特征 | 原置信度选1个关节、完全替换；后续GCN不改 | 样本相关替换是否优于固定模板？ |

B第一版用共享小门控MLP（256→32→1，sigmoid），输入选中关节特征，末层零初始化令初始alpha=0.5。不增加额外监督或损失。D先用单层256→256投影；固定邻接表来自COCO17点骨架，不学习选邻居；没有邻居的情况回退原模板。新增参数和计算都须报告，不能声称自动压缩参数或跳过图计算。

每人用独立模块/类与独立配置，仅在统一配置基础上覆盖自己的CIM字段。公共接口保持输入[B,17,256]，返回同形状特征和[B,17]布尔mask。禁止直接把四版混在一起。

### 实现检查

- 原选择逻辑在`torch.no_grad()`内：B门控、D投影及其特征混合必须在可导区域，不能一起放入no_grad。
- 仍返回布尔mask给现有一致性损失，不改损失定义。
- forward、loss、predict路径都使用同一CIM；不只改训练路径。
- 不使用GT关键点/可见性作为推理输入；无额外数据或标签。
- 检查新增参数非空梯度、有限loss、FP16正常，以及batch128峰值显存。门控有梯度不等于整个初始置信度分支都得到监督，不擅自修改该分支训练方式。
- 至少记录alpha分布（B）、被选关节分布（C）、替换特征范数（D），用于排查退化；这些诊断不能用来反复挑验证集最优设置而不披露。

## 快速实验流程

1. 原版CIM baseline128已完成100轮，使用results中的第10/20轮AP进行初筛。
2. 四人各自做短程运行检查，然后从预训练骨干跑到第20轮。保持100轮调度，不改成20轮余弦；相同第10/20轮AP做初筛，20轮不能证明最终收敛。
3. 选1–2个候选从各自第20轮checkpoint完整恢复到100轮，保留优化器、EMA和调度状态；最终比较同轮AP、最佳AP与其轮次。
4. 若时间允许，对胜出版本及baseline增加共同随机种子复核；单seed微小提升只能当作初步证据。

仓库根目录，激活自己的CIGPose环境后：

```bash
# 原版baseline：100轮
python lightweight/scripts/train_team128.py --work-dir work_dirs/team128_baseline

# 候选20轮初筛：将配置替换为成员自己的继承配置
python lightweight/scripts/train_team128.py --config lightweight/configs/member_a.py --work-dir work_dirs/team128_a --stop-after-epoch 20

# 同一候选从20轮继续至100轮
python lightweight/scripts/train_team128.py --config lightweight/configs/member_a.py --work-dir work_dirs/team128_a --resume work_dirs/team128_a/epoch_20.pth
```

`member_a.py`是示例文件名，须成员实现后提供，不是本次已交付的CIM实现。后台运行可在命令前加`nohup`并重定向到独立日志；不要同时在同一张卡启动多份任务。runner只锁定同一work-dir，不能阻止其他目录的并行占卡。首次运行要求新结果目录，防止覆盖。

## 每人交付

代码分支和commit、唯一改动说明、完整配置、依赖版本及GPU型号、seed、训练日志、验证JSON、最佳权重路径、训练总时长、参数量和峰值显存。模型权重不放Git仓库，另外共享文件。

| 版本 | commit/环境 | AP@10 | AP@20 | AP@100 | 最佳AP/轮次 | 相对同环境baseline增益 | 参数量 | 时间/显存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 原版CIM128 | 5090，统一配置 | 65.1766 | 67.2113 | 70.3364 | 70.3364 / 100 | — | 6,190,372 | 约4小时30分 |
| A固定混合 | 待填写 | | | | | | | |
| B可学习强度 | 待填写 | | | | | | | |
| C熵选择 | 待填写 | | | | | | | |
| D上下文替换 | 待填写 | | | | | | | |

不把日志acc_pose当作COCO AP，不用跨epoch AP平均值代替标准验证AP。优先比较B对A的增益，才能区分学习门控与简单减弱干预的收益。
