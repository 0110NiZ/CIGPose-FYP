# 公共精度协议 V2

2026-11-16：保留旧 TF32 精度，仅开启全局确定性算法。用户已批准；不训练基础模型，不覆盖历史缓存。

统一入口：mmpose/analysis/fyp/precision_policy.py。在每个 GPU 程序开始、CUDA 初始化之前调用 configure_legacy_precision(seed=42, threads=4)；assert_legacy_precision() 检查状态，runtime_record() 保存来源。

| 设置 | 值 |
|---|---|
| profile_id | cigpose-legacy-cache-fp32-tf32conv-deterministic-v2 |
| 模型 | FP32，无 autocast/FP16/BF16 |
| CUDA matmul TF32 | False |
| cuDNN TF32 | True |
| cuDNN benchmark | False |
| cuDNN deterministic | True |
| 全局确定性算法 | True |
| CUBLAS_WORKSPACE_CONFIG | :4096:8，CUDA 初始化前设置 |
| 基础模型 | 冻结、eval、single-view、flip_test=False |
| 输入和解码 | 192×256、SimCC split_ratio=2、use_dark=False |

公共缓存生成和前向检查入口已接入；候选接口拒绝不一致精度及超过 32 人的批次。保留原缓存批次和顺序，17 个干预从原特征独立生成，不合并为大批次。该限制针对候选推理，不是 MLP 训练 batch size。

历史 train/val 缓存没有修改或补写来源字段；新缓存写入 profile。不要用新版生成器续写历史缓存，应使用新目录。基础 head、checkpoint 和评估规则不变。B/C 的新程序也必须调用此初始化，不能依赖另一个进程的设置。

正式收益使用固定缓存 P0，重算 P0 仅作诊断，不要混用。原图抽查 285 人的特征和 uncertainty 与历史缓存逐值一致，不代表所有重算 P0 完全一致。确定性不保证跨设备、软件版本或批次布局逐位一致。

严格旧开关版本复验仍有少量不稳定候选，因此最终使用 V2。历史 AP/AR 不直接标为 V2 新结果；正式比较前按当前协议复核基线。

公共设置、入口适配和基线重跑代码在 public-tasks；A 的实现及结果在 feature/module-a。标签、缓存、图片和权重不纳入 Git。当前公共基线见 experiments/public_tasks/baselines_v2_20261116/README.md。
