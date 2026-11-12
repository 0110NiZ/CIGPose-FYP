# N 的 2C：共享参考缓存

基于 T 的 `feature/intervention-interface` / `6fd5b3c`。本实现不训练 ABC，
不生成 17 个关节或四种强度的收益标签，不更改基础模型参数。

## 1. 文件与职责

- `mmpose/analysis/fyp/build_shared_cache.py`：真实模型生成入口、解码、身份校验、加载后候选一致性检查。
- `mmpose/analysis/fyp/shared_cache.py`：NumPy 缓存读写、原子保存、续跑、字段检查、train/val 交集检查。
- `tests/test_shared_cache.py`：不依赖 PyTorch 的存储和数据契约测试。

生成入口复用 T 的 `head.extract_refinement_state()`、
`head.cim.apply_intervention()`、`head.predict_from_joint_features()`。
注意：apply_intervention 位于 head.cim，不是 head 本身。
P0 用 head.decode 和 model.add_pred_to_datasample 转成原图坐标，保留原解码分数。
重载候选用相同 codec 和 top-down 坐标公式，不用手写 argmax 替代 codec。

## 2. 使用 L 已交付的全量有效人体清单

已接入 L 的 `lin` 分支提交 `0371b35`：train2017 共 149,813 人，val2017 共 6,352 人。
清单及统计保存在 `manifests/`。省略 `--valid-persons` 时按 `--split` 自动选择新清单。
程序要求 `supervision_filter_v2`，不会使用旧 90/10 划分。两份清单格式如下：

```json
{
  "dataset_split": "train2017",
  "source_annotation": "person_keypoints_train2017.json",
  "source_annotation_sha256": "完整原始标注文件的实际 SHA-256",
  "filter_version": "supervision_filter_v2",
  "valid_ann_ids": [123, 456]
}
```

以上 ID 只是格式示例，不是可运行的真实名单。val 使用 `val2017` 和相应文件名。
SHA-256 可用 PowerShell `Get-FileHash -Algorithm SHA256` 获取（填写小写结果）。
必须是实际完整清单；程序检查所有列出的人体合法且确属该标注文件，检查数据集
没有静默丢人，但不替 L 证明名单已包含所有有效人体。全量覆盖需与 L 的过滤报告核对。
原始标注中每个人体的 image_id 是身份依据，不靠 DataLoader 行号。

## 3. 环境

在 T 已运行 smoke test 的 CIGPose 环境运行，需该仓库的 MMPose、PyTorch、
MMEngine/MMCV 和 COCO 数据依赖。NumPy-only 的缓存检查和单元测试不需 GPU。
必须使用协议中的 checkpoint SHA-256；路径可换，文件内容不可换。
GT-box 缓存只用于离线监督/开发，不能冒充检测框 benchmark。缓存没有 NMS 或 AP 计算。

命令从仓库根目录执行；以下使用正斜杠，Windows/Linux 均可按实际路径替换。

## 4. 先生成 smoke 缓存

```text
python mmpose/analysis/fyp/build_shared_cache.py --split train2017 --ann data/coco/annotations/person_keypoints_train2017.json --images data/coco/train2017 --valid-persons manifests/train2017_valid_persons.json --checkpoint checkpoints/cigpose-m_coco_256x192.pth --output outputs/cache/smoke_train --max-instances 8 --batch-size 4 --device cuda:0
```

每次新运行/续跑的第一个新 batch 都自动检查：

1. 特征转为 float32 NumPy，保存并重载。
2. 从重载特征执行 alpha=0，经 GNN 和 codec 恢复 P0。
3. 从重载特征执行混合 joint/alpha，与实时路径生成的候选原图坐标和分数比较。
4. 每个 shard 写盘后逐字段精确回读比较，再原子发布。

这些检查并不替代 Y 的独立可视化、GT 对齐、alpha=1 和 OKS 验证。
`--max-instances` 生成的缓存明确标为 smoke，必须与全量输出分目录；不能拿它
证明全量有效性。第一次生成前先运行 T 的 smoke tests，并由 Y 核验真实样例。

## 5. 全量和续跑

```text
python mmpose/analysis/fyp/build_shared_cache.py --split train2017 --ann data/coco/annotations/person_keypoints_train2017.json --images data/coco/train2017 --valid-persons manifests/train2017_valid_persons.json --output outputs/cache/train2017 --batch-size 32 --device cuda:0
python mmpose/analysis/fyp/build_shared_cache.py --split val2017 --ann data/coco/annotations/person_keypoints_val2017.json --images data/coco/val2017 --valid-persons manifests/val2017_valid_persons.json --output outputs/cache/val2017 --batch-size 32 --device cuda:0
```

中断后使用**同一条命令加 `--resume`**。程序读取实际 shard 恢复完成的 annotation ID，
不会只相信进度数字，也不会重新前向已经保存的人体。尾部未发布 `.tmp` 文件被忽略。
权重、展开配置、数据/名单、实现代码、协议、batch size 或设备变化都会拒绝混用。
不要在续跑中升级环境或改图片内容。更改这些条件时用新目录重新生成。

一次只有一个写入进程。正常退出及 Ctrl+C 会释放 `.writer.lock`。强制结束进程或
断电可能留下锁：确认没有任何进程写该目录后，才可手动移除该锁并 `--resume`；
不要删除已经提交的 shard。进度 `status.json` 是参考，实际完整性由 shard 决定。

## 6. 缓存格式

目录含 `run.json`（协议、指纹和完整预期 image/annotation 对）、`status.json`、
`shard-*.npz`。每个 shard 默认是一个生成 batch。NPZ 不使用 pickle，浮点为 float32，
ID 为 int64，按 shard 读取避免一次加载全部特征。

| 字段 | 每人体形状 | 含义 |
|---|---|---|
| features | 17×256 | GAU 后、干预前的关节特征 |
| uncertainty / confidence | 17 | 初始分布的不确定性/置信度，不是最终分数 |
| top1_joint | 标量 | 不确定性最高的关节 |
| P_0_coords / P_0_scores | 17×2 / 17 | 原图坐标与 codec 分数 |
| image_id / annotation_id | 标量 | 原始 COCO 身份；instance_id 可由 annotation_id 转为字符串 |
| bbox / bbox_score | 4 / 标量 | 经数据集裁剪的 xyxy 框及框分数 |
| input_center / input_scale / input_size | 各 2 | 用于原图坐标恢复；input_size 按 width,height |
| original_size | 2 | 原图 width,height |

dataset_split 在 run.json 中，由读取器提供。`features` 和缓存状态不含 GT 坐标、
GT area、OKS 或 gain。生成目录中的 `selected_annotations.json` 是供数据集构建的
原始标注辅助文件，属于离线监督资料，不是模型输入！标签生成者按 annotation_id
从标注取得 GT，禁止将整个标注或 run.json 的身份字段拼进学习特征。

## 7. 检查和读取

```text
python mmpose/analysis/fyp/shared_cache.py outputs/cache/train2017
python mmpose/analysis/fyp/shared_cache.py outputs/cache/train2017 --val-cache outputs/cache/val2017
python -m unittest discover -s tests -p test_shared_cache.py -v
```

未完成目录加 `--allow-partial` 只检查现有 shard，不把它视为全量完成。
train/val 联合检查要求两者完整，并拒绝图片 ID 和人体 ID 交集。

```python
# 从仓库根目录运行，或在模块中按你们的包布局导入。
import sys
sys.path.insert(0, "mmpose/analysis/fyp")
from shared_cache import ReferenceCache

cache = ReferenceCache("outputs/cache/train2017")
for batch in cache.iter_shards():
    features = batch["features"]
    annotation_ids = batch["annotation_id"]
    # 只将约定的输入字段传给模块；按 ID 与离线标签对齐。
```

## 8. 本机验证边界

2026-11-12：14 项 NumPy 单元测试通过；在本机 WSL Ubuntu-22.04 的 `cigpose`
环境（PyTorch 2.0.1+cu118、RTX 4060 Laptop）完成 train/val 各 8 人真实缓存验证。
保存重载后的 P0 和混合 joint/alpha 候选与实时路径一致；16 人的 alpha=0 恢复、
alpha=1 候选、人体身份及 L 的 OKS/gain 与 COCOeval 对照均通过。
同时对完整原始标注复核 L 的文件字节 SHA-256 和过滤结果，全部匹配；完整
train/val 图片 ID 无交集。
WSL 缓存及报告目录：`/home/nzd/projects/cig-shared-cache-smoke-20261112/`。

同日完成全量生成：train2017 149,813 人、val2017 6,352 人，batch size 32、
2 个数据加载 worker。所有分片逐一回读校验通过，无遗漏或重复人体，
train/val 图片 ID 和 annotation ID 交集均为 0。总耗时约 27.7 分钟。
正式输出位于 WSL `/home/nzd/projects/cig-shared-cache-run-20261112/`：
`train2017/`、`val2017/`、各 split 的 `.log` 和 `full_validation_report.json`。
`code/` 保存本次运行的独立代码副本。全量缓存保持 single-view、冻结 eval 和 GT 框协议；
它是离线模块开发输入，不是最终检测框 AP/AR 评测结果。

第一版只接受当前确定性的四步验证 pipeline，禁止随机翻转、随机裁剪及隐式训练增强。
改变增强策略意味着不同输入，需要单独协议及缓存版本。单视图缓存不提供翻转分支。
