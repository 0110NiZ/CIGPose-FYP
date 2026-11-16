# CIGPose FYP：当前状态与 A 模块训练、验证方案

更新日期：2026-11-16。本文依据本机公共任务记录、当前代码和 V2 标签验收结果整理。

目录更新：实际可编辑的 MLP、参数和训练入口已集中到仓库根目录 `Module_A/`。旧 `train_a.py`、`gain_predictor.py` 及其 WSL 字节码缓存已清理，必要的公共依赖保留。原来生成的 Windows 阅读包和 V2 交付 ZIP 是历史快照，不会自动随新目录更新；新代码以本分支为准。

**当前结论：可以开始全量 MLP 训练的下一阶段，但本次只整理方案，没有启动训练。** 标签验收通过代表数据可用于约定协议下的训练，不代表 A 已提升 AP。

## 1. 项目要做什么

基础 CIGPose 负责提供人体姿态、关节特征和 uncertainty。新增模块分别解决：

| 模块 | 决策 | 首轮单模块实验中固定的条件 |
|---|---|---|
| A | 选哪个关节重新计算 | α=1，始终接受候选 |
| B | 使用多大的干预强度 α | 固定不确定性 Top-1 关节，始终接受 |
| C | 接受候选还是回退 P₀ | 固定不确定性 Top-1，α=1 |

先验证各模块，再组合。当前没有必须解冻基础 CIGPose 的需求；MLP 训练只更新 A 的参数。以后若改变基础模型权重，原特征缓存和收益标签可能失效，必须重新检查或生成，属于独立实验。

## 2. 已确定的数据协议

| 阶段 | 数据 | 输入框 | 用途 |
|---|---|---|---|
| 训练 | 完整 train2017 有效人体 | GT 框 | 生成收益标签、训练 A/B/C |
| 验证 | 完整 val2017，5,000 张图 | 固定检测框，104,125 条 | 调参、选模型、消融、AP/AR |
| 最终测试 | test-dev | 检测框 | 模型和设置锁定后最终评估 |

train 使用 GT 框可以直接建立人体与 GT 的监督对应，并复用已验收缓存。使用检测框训练可以作为以后独立扩展，需要 train 检测框、匹配规则和新的标签，不是当前训练的前置条件。

val 的“检测框”是模型输入；val 的“GT”是评估答案，两者不矛盾。正式评估保留全部约定检测实例，由统一 COCO 评估入口完成匹配；不能先用 GT 删除误检或只评估检测成功的人。漏检和误检会影响最终精度/召回。

GT 坐标、真实 Gain 和 GT 人体面积不进入 A 的预测输入。val 不参与梯度更新，也不参与输入标准化统计。test-dev 不用于调参。

## 3. 公共任务与历史基线

公共任务 2B～5 已完成检测框前向、缓存、联调、统一评估、全量缓存和基线验收。公共分支为 public-tasks，基础提交为 66edb12f7e1f8efc83ca4123f8ee67bda4bf68dd。

| 数据 | 人体/检测数 | 分片数 |
|---|---:|---:|
| train GT 缓存 | 149,813 | 4,682 |
| val detection 缓存 | 104,125 | 3,254 |

历史结果采用 0～1 标度：

| 方法 | AP | AR |
|---|---:|---:|
| 原始 CIGPose | 0.735402168 | 0.789704030 |
| 不干预 P₀ | 0.736222993 | 0.789767003 |
| 不确定性 Top-1 + α=1 | 0.735402168 | 0.789704030 |

Original 和 Top-1 的汇总指标相同，但逐项预测不完全相同。以上是历史公共实验结果，不能直接标为本次确定性 V2 设置的新结果。正式报告 A 增益前，应在当前统一协议下复核对照；涉及候选推理的基线应重新生成并评估。固定缓存 P₀ 可直接用相同缓存输出和评估规则复算。

## 4. 当前统一精度设置

正式 profile：`cigpose-legacy-cache-fp32-tf32conv-deterministic-v2`。

| 项目 | 设置 |
|---|---|
| 模型数据类型 | FP32，不启用 autocast/FP16/BF16 |
| CUDA 矩阵乘法 TF32 | False |
| cuDNN TF32 | True，保留旧卷积精度 |
| cuDNN benchmark | False |
| cuDNN deterministic | True |
| 全局确定性算法 | True，用户批准的调整 |
| CUBLAS_WORKSPACE_CONFIG | `:4096:8`，CUDA 初始化前设置 |
| 默认随机种子 / CPU 线程 | 42 / 4 |
| 基础 CIGPose | 冻结、eval、single-view、flip_test=False |
| 输入 / 解码 | 192×256、SimCC split_ratio=2、use_dark=False |

公共初始化由 `precision_policy.py` 提供。本机缓存和前向相关入口、A 的入口已经接入；B/C 后续需要调用同一入口。其他成员电脑和 GitHub 没有自动更新。

候选计算保持原缓存批次和顺序，单批最多 32 人，17 个干预独立从原特征生成。**MLP 训练 batch_size=256 是另一件事**：它只读取已有输入和标签训练小网络，不重算 CIGPose 候选，因此不受候选批次 32 的同一限制。

确定性设置是本环境复现协议，不承诺跨 GPU、驱动和软件版本逐位相同。

## 5. 标签版本与验收结果

### 5.1 为什么重生成

Y 原包包含 149,813 人、2,546,821 个 Gain。后续发现重复推理的数值问题，需要统一运行协议并核验，而不是直接认为其全部标签错误。

严格旧开关下重新生成的 V1，在 50,592 个复验候选中仍有 9 个超出验收容差，因此不用于正式训练。

用户随后批准：保持旧 TF32 精度，仅开启全局确定性算法。按此生成独立 V2；Y 原包、V1 和公共缓存全部保留。

### 5.2 正式 V2 标签

每个人对应 17 个标签：

`gain[j] = OKS(Pj, GT) - OKS(原缓存 P₀, GT)`

其中 Pj 是仅选择第 j 个关节、α=1 时的完整姿态候选。Gain 表示**整个人体姿态 OKS 的变化**，不是该关节自身置信度，也不是 AP。

- 全量 4,682 分片、149,813 人、2,546,821 个 Gain；全量文件、ID、覆盖率、公式及缓存 P₀ 的 OKS 检查通过。
- 独立 GPU 复验 95 分片、3,040 人、51,680 候选。
- 最大复验 OKS 差约 4.44×10⁻¹⁶，最佳关节变化 0。
- 平均 Gain 为 -0.010145525813061869。不是所有干预都有益，保留负值；平均为负不直接决定最优选择策略是否有效。
- 标签读取、真实数据接口、32 项回归测试通过。
- 没有训练，优化器更新次数为 0。

### 5.3 尚需明确保留的数值边界

抽查 285 个原图实例，特征和 uncertainty 与旧缓存逐值一致；未重建全部公共缓存。

重算 P₀ 与旧缓存仍存在少量差异：最大坐标差 2.05279541015625 像素、最大 OKS 差 0.0020611701345323308，27 人的 OKS 差超过 1e-5。因此正式标签始终相对于**固定原缓存 P₀**计算，不声称历史缓存能逐位重算还原。

对同一个人，减去共同的 P₀ OKS 常数不会改变 17 个候选的真实收益排序；但会影响收益的绝对值和正负判断，尤其关系到后续 C 的接受/回退。因此 A/B/C 必须明确共用哪个 P₀，不能混用重算 P₀ 与缓存 P₀。

最终验收状态以 `acceptance_report.json` 的 passed:true、`metadata.json` 的 complete:true 和 `reader_check.json` 为准。`generation_summary.json` 中 accepted:false 是验收前生成阶段的记录。

## 6. 当前 MLP 实际怎样设计

当前实现约 67,281 个参数，输出每个人的 17 个预测收益。

| 输入 | 每人的形状 | 含义 |
|---|---|---|
| features | 17×256 | 基础 CIGPose 的关节特征 |
| uncertainty | 17 | 各关节的不确定性 |
| pose_coords | 17×2 | P₀ 预测位置，按输入中心和尺度归一化 |
| pose_scores | 17 | P₀ 关键点预测置信度，不是检测框分数或 Gain |

当前网络对每个关节读取：该关节 256 维特征、1 个 uncertainty、16 维关节身份 embedding，以及 64 维全身上下文。

全身上下文由所有 17 个关节的 x、y、置信度共 51 个数，经 `51→128→64` 的 MLP 得到。收益回归网络为 `337→128→64→1`，在 17 个关节间共享参数。

**所以当前 MLP 已经能看到其他关节的位置和置信度；但还没有将所有关节的 features 和 uncertainty 一起放入全局上下文。** 全关节特征融合、全局 uncertainty 或 Transformer 是后续独立结构消融，不在第一轮同时修改。

末层没有 sigmoid/softmax，允许输出正负收益。推理选择 `argmax(17 个预测收益)`；即使全部为负，A-only 仍选一个并接受，回退属于 C 的职责。

## 7. 第一轮具体怎么训练

这是一套拟执行方案，尚未运行。直接使用全部有效 train 数据，不先做缩小数据规模的研究实验。

1. 固定 V2 标签、train 缓存、val 缓存、checkpoint、代码和精度配置的指纹，使用新的训练输出目录。
2. 仅从 train 估计 uncertainty 与关键点分数的均值/标准差，保存在模型中；特征使用 LayerNorm，坐标按人体输入尺度归一化。
3. 每轮打乱分片，并在有界缓冲区内打乱人体；读取一个 batch 的输入以及按 annotation_id/image_id 对齐的 17 维真实 Gain。
4. MLP 输出 `[batch,17]` 的预测 Gain。
5. 使用 `SmoothL1(预测Gain×100, 真实Gain×100)` 计算误差，反向传播只更新 MLP。
6. 一轮完整覆盖 149,813 人，检查无遗漏和重复；保存训练 loss 和 Gain MAE。
7. 按固定间隔对完整 val 检测缓存运行 A-only，按 val AP 保存 best.pt，并保存 last.pt 支持恢复。

当前代码默认值，作为第一轮起点而非已经调优的结论：

| 参数 | 第一轮设置 |
|---|---|
| Epoch | 20 |
| MLP batch size | 256 |
| 优化器 | AdamW |
| 学习率 | 0.001，当前代码为固定学习率 |
| Weight decay | 0.0001 |
| 损失 | SmoothL1，预测和目标同时乘 100 |
| 梯度裁剪 | norm 5 |
| 验证频率 | 每 5 轮以及最后一轮，即 5/10/15/20 |
| 第一轮 seed | 42 |
| 最佳模型标准 | 完整 val2017 的 COCO AP 最大 |

训练期无需每个 batch 再跑 17 次 CIGPose，标签已经离线生成。完整 val 评估仍要为选中关节计算候选，可能比小 MLP 的参数更新更耗时；未做实测，不预报固定训练时长。

训练入口默认仅 preflight，只有显式 `--run-training` 才更新参数。本次没有传入该开关。

## 8. 如何验证 A 是否提升

### 8.1 最重要的受控对照

所有组必须使用相同 base checkpoint、val 检测文件、缓存、解码、精度、评分、NMS 和完整 5,000 张图。

| 对照组 | 关节选择 | α | 是否接受 | 回答什么问题 |
|---|---|---|---|---|
| P₀ | 不干预 | — | — | 干预后是否优于原始起点 |
| 不确定性 Top-1 | 既有固定策略 | 1 | 始终接受 | A 的主要对照 |
| A-only | MLP 预测收益最大 | 1 | 始终接受 | 学习选择是否更好 |
| 原始 CIGPose | 原实现 | 原实现 | 原实现 | 与原方法的整体比较 |
| 随机关节（补充） | 固定随机种子均匀选择 | 1 | 始终接受 | 提升是否只是随便换一个关节 |

主要计算：

- `ΔAP_selection = AP(A-only) - AP(不确定性Top-1)`：选择策略的收益。
- `ΔAP_net = AP(A-only) - AP(P₀)`：相对于不干预的净收益。

例如，若 A 比 Top-1 好但仍比 P₀ 差，只能说“改善了干预的关节选择”，不能说“干预整体已优于不干预”。若改动 α、接受规则、检测框或后处理，则不能把全部变化单独归因于 A。

AP 在文件中使用 0～1 标度；报告百分点时乘 100。例如 0.735→0.737 是 +0.002，即 +0.2 AP 点。

### 8.2 不只看 loss

主指标是完整 val 的 AP，同时保存 AR、AP50、AP75、APM、APL 等公共指标。训练 loss 下降或训练 Gain MAE 变小，不足以证明选关节有效；单人 OKS 收益也不等同于最终 AP，因为评分排序和 NMS 会影响结果。

利用每次验证保存的 selections.jsonl 分析：17 类关节选择频率、是否总选同一关节、与 Top-1 的选择一致率、代表性改善/退化样例。

可选的进一步机制分析：在固定规则匹配到 GT 的 val 人体上，单独生成 17 候选的诊断收益，比较所选真实 Gain、负收益率、距离最佳候选的损失（regret）。这仅用于解释机制，不能提前用 GT 筛掉正式 AP 输入，不能把 val 诊断标签用于训练。这部分当前尚未完成，不是开始首轮训练的必需条件。

### 8.3 判断提升是否稳定

第一轮 seed=42 用于观察是否学到信号。若结果有希望，在相同数据、结构和超参数下补做例如 42/43/44 三个训练种子，每次按相同 val 选模规则，报告平均 ΔAP、标准差和各次结果，不只挑最好的一次。

若提升极小或不同种子方向不一致，先报告“不稳定”，不能直接写成确定提升。需要更严格结论时，可补充按图像配对重采样的 AP 差异分析，保留每张图的全部检测并正确运行评估；当前没有实现或运行这项分析。

val 用于选模和调参，因此最终泛化结论仍应在设置锁定后由 test-dev 检验，不能把反复调过的 val 当成完全独立测试集。

## 9. 第一轮结果不好时怎么办

先检查标签/输入身份、选择分布、训练和 val 曲线、GT 框训练到检测框验证的差异，区分“学不会”“只学会训练集”和“学会收益但 AP 不提升”。

保持对照不变，每次只改一个因素。候选改动包括加入所有关节的 uncertainty/特征上下文、增加排序损失、换 Transformer。先做当前 MLP 基线，才能说明后续改动究竟改善了什么。

若考虑“预测收益小于零时不干预”，这是引入接受/回退策略，应独立标为 A+C 类实验，不能悄悄加入 A-only。

## 10. 分工和交接

| 人员 | 当前主要职责 |
|---|---|
| N（你这边） | 公共精度统一、正式 V2 标签重生成与验收已完成；下一阶段负责 MLP 全量训练、保存模型及 A-only 评估 |
| Y | 原候选/标签生成工作；接收 V2 协议与结果后协助标签审计、选择策略分析、可视化和错误分析 |
| L | B，复用公共候选接口和同一精度/P₀ 定义 |
| T/M | C、候选接受回退和统一 AP/AR，保持实例对应及同一 P₀ 定义 |

目前不必等待 Y 再提供新标签才能开始训练；已验收的 V2 是当前正式训练输入。成员间仍需同步版本，避免一方用 Y 原包、另一方用 V2 却当作同一个实验。

## 11. 文件和路径

### WSL 实际工作路径

| 用途 | 路径 |
|---|---|
| Python 环境 | `/home/nzd/miniconda3/envs/cigpose/bin/python` |
| 当前代码 | `/home/nzd/projects/cig-module-a-n/code` |
| 日常使用的 A 目录 | `/home/nzd/projects/cig-module-a-n/code/Module_A` |
| train GT 缓存 | `/home/nzd/projects/cig-shared-cache-run-20261112/train2017` |
| val detection 缓存 | `/home/nzd/projects/cig-detection-public-20261116/val2017-detection` |
| 正式 V2 标签 | `/home/nzd/projects/cig-module-a-n/labels/train2017_legacy_deterministic_v2_20261116` |
| 基础权重 | `/home/nzd/projects/CIGPose/checkpoints/cigpose-m_coco_256x192.pth` |
| COCO 数据 | `/home/nzd/projects/CIGPose/data/coco` |
| V2 接口检查 | `/home/nzd/projects/cig-module-a-n/checks/legacy_deterministic_v2_20261116` |

WSL 当前代码是本地快照，没有 .git。Windows 项目中的 CIGPose-FYP 用于 Git 交付；公共精度设置和 A 模块已分开提交，本次继续把目录整理及清理作为后续提交发布到 feature/module-a，具体提交号以分支历史为准。

### Windows 交付

交付根目录：

`C:/Users/15790/.codex/.chatgpt-projects/g-p-6af31ba6cf8081918fce4ec80eda9839/output/legacy_precision_rollout_20261116/`

- `delivery/Module_A_Legacy_Deterministic_V2_Labels_20261116.zip`：正式标签包。
- `delivery/Public_Precision_Legacy_Deterministic_V2_20261116.zip`：公共代码覆盖文件、说明和检查结果；在公共代码基础上使用，不是完整仓库。
- `delivery/SHA256SUMS.txt`：压缩包校验值。
- `验收完成报告.txt`、`公共旧精度使用说明.txt`：结果与协议。
- `preflight_v2.log`、`interface_check_v2.json`、`tests_v2.log`：读取、接口、测试记录。
- `before_windows/`、`before_wsl/`、`before_deterministic_adjustment/`：修改前备份。

Y 原始包仍位于 `C:/Users/15790/Desktop/Module_A_Train2017_Labels.zip`，校验值与验收记录一致。旧缓存和失败 V1 也未删除。

常用代码在 `Module_A/`，公共依赖在 `mmpose/analysis/fyp/`：

| 文件 | 功能 |
|---|---|
| Module_A/model.py | 当前 MLP，类名 MLP |
| Module_A/config.py | 路径与训练参数 |
| Module_A/data.py | 复用公共数据读取器 |
| Module_A/train.py | 默认预检查，显式开启才训练 |
| Module_A/evaluate.py | 单独完整 val 评估 |
| mmpose/analysis/fyp/precision_policy.py | 统一精度和确定性设置 |
| mmpose/analysis/fyp/build_a_labels_legacy.py / verify_a_labels_legacy.py | V2 标签生成/验收 |
| mmpose/analysis/fyp/a_contract.py / a_dataset.py | 输入白名单、标签匹配、全量读取 |
| mmpose/analysis/fyp/predict_a.py | A-only 候选生成和统一评估辅助函数 |

在 WSL 激活 cigpose 后，进入 `/home/nzd/projects/cig-module-a-n/code/Module_A`，运行 `python train.py` 只预检查，确定训练时才加 `--run-training`。默认输出目录为 `/home/nzd/projects/cig-module-a-n/runs/module_a_mlp_seed42`。运行方法见本目录 README。

清理后重新通过 18 项 A 测试、6 项精度测试，以及 train 149,813 / val 104,125 的全量预检查；检查没有优化器更新。记录见 `experiments/module_a/layout/`。

## 12. 目前未做与下一步顺序

目前未做：MLP 参数训练、A 的正式 AP/AR、跨种子训练、B/C 训练、组合实验、test-dev 最终测试。没有证据可以宣称 A 已提升 AP。

test-dev 图片、清单、检测框和固定 val 检测器上游配置/权重，按现有公共记录仍待补齐；本次没有重新搜索这些资料。最终测试需要同源检测协议，若更换检测器，要重跑相应 val 对照。

下一阶段建议顺序：

1. 固定本次 V2 配置和文件版本，复核当前协议下的 P₀、Top-1 等对照结果。
2. 使用全量 train 启动当前 MLP，先运行预先约定的 20 轮、seed=42；基础 CIGPose 不训练。
3. 每 5 轮完整 val 评估，选择 best.pt，整理与 Top-1、P₀ 的 AP/AR 差和关节选择分布。
4. 有信号后补充重复种子与必要的消融；若无信号，先分析再单独改设计。
5. 单模块结论成立后推进组合，最后锁定模型进入 test-dev。

本次仅核对代码并生成这份总结；没有运行以上训练或新的全量评估。
