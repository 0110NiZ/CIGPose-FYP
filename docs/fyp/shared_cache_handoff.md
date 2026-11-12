# N 的 2C 共享缓存交接

## 状态与来源

2026-11-12 完成，分支 `feature/shared-cache`。

- T 公共接口基于 `6fd5b3c`；未修改 T 的模型实现。
- L 清单、过滤脚本及 OKS 工具原样取自 `0371b35`，不依赖旧 90/10 划分。
- 本分支提供缓存生成/读取程序、14 项单元测试、全量人体清单及验证报告。
- 已在本机 WSL Ubuntu-22.04，`/home/nzd/miniconda3/envs/cigpose/bin/python`
  环境运行（PyTorch 2.0.1+cu118，RTX 4060 Laptop）。
- 主协议：single-view、flip_test=False、冻结模型并 eval、GT 框、float32。

## 已完成验证

| 内容 | 结果 |
|---|---|
| train2017 全量 | 149,813 人，约 2.9 GiB |
| val2017 全量 | 6,352 人，约 125 MiB |
| 缓存完整性 | 所有分片回读通过，无遗漏/重复，字段及身份映射通过 |
| train/val 交集 | 图片 ID 与人体 annotation ID 均为 0 |
| 原始 COCO 标注 | 原始文件 SHA-256 与 L 清单一致，重跑过滤得到相同完整名单 |
| 真实模型联调 | train/val 各 8 人，缓存重载 P0 和混合干预候选匹配实时路径 |
| 真实 OKS 联调 | 16 人的 P0/Pc 与 COCOeval 计算差异为 0；alpha=0 最大绝对 gain 约 2.31e-7 |

报告：`experiments/shared_cache/full_validation_report.json`、
`experiments/shared_cache/real_oks_smoke_report.json`。
全量生成总耗时约 27.7 分钟，batch size 32，workers 2。
真实候选和 OKS 联调是 16 人测试；未给全部人体生成候选收益标签。

## 缓存在哪里

大体积缓存、图片、权重没有上传 GitHub。缓存目前仅在 N 的本机 WSL：

```text
/home/nzd/projects/cig-shared-cache-run-20261112/
  code/                         # 生成时固定的代码副本
  train2017/                    # run.json、status.json、shard-*.npz 等
  val2017/
  train2017.log
  val2017.log
  full_validation_report.json
```

小规模缓存位于 `/home/nzd/projects/cig-shared-cache-smoke-20261112/`。
接手人使用其他机器时，需要从 N 复制 train2017/、val2017/ 两个完整目录，
或者根据 `shared_cache.md` 重新生成。仅拉取 GitHub 分支不会下载这些缓存。
完整复制后可以使用下方只读命令验收，无需重新前向或重新生成。

## 接手顺序

1. 拉取并切换到本分支：

   ```bash
   git fetch origin
   git switch --track origin/feature/shared-cache
   ```

2. 在仓库根目录运行存储测试和缓存验收（只需 NumPy）：

   ```bash
   python -m unittest discover -s tests -p test_shared_cache.py -v
   python mmpose/analysis/fyp/shared_cache.py /home/nzd/projects/cig-shared-cache-run-20261112/train2017 --val-cache /home/nzd/projects/cig-shared-cache-run-20261112/val2017
   ```

   其他机器把路径改成复制后的实际位置。

3. 按 `shared_cache.md` 的读取示例逐分片读取；按 annotation_id 连接 GT/标签，
   不要按行号拼接。GT、true OKS、true gain 只用于离线标签与验证，不能作为模块推理输入。

4. 开始模块候选与标签生成：A 在固定 alpha=1 下比较 17 个关节；
   B 使用约定关节选择方式比较 0.25/0.50/0.75/1.00；C 根据指定 joint/alpha 的
   候选构建接受标签。每个候选从相同原始 features 独立生成。

5. 统一使用 `oks_utility.compute_gain(P_c, P_0, gt_kpts, area)`，
   即 `OKS(P_c, GT) - OKS(P_0, GT)`。不要混入早期误差差值标签。

## 运行与复现边界

- 缓存已有 features、uncertainty、confidence、top1_joint、原图坐标 P0/score、
  bbox 和坐标恢复参数；它不含 A/B/C 的候选或训练标签。
- `selected_annotations.json` 是数据构建的离线 GT 辅助文件，不能整体作为模型输入。
- 原始数据在 `/home/nzd/projects/CIGPose/data/coco/`；checkpoint 在
  `/home/nzd/projects/CIGPose/checkpoints/cigpose-m_coco_256x192.pth`。
- 程序支持续跑，但严格比较代码、配置和协议。已经生成的缓存可直接读取；
  不要在新的 Git commit/不同代码副本上对旧目录盲目追加。若确需恢复旧运行，
  使用保留的 `code/` 副本和原始参数。
- 正式主实验已统一为 GT 框、single-view，配置与 M 的评估任务见 `evaluation.md`。
  缓存无需重建，但不是 AP/AR 成绩；导出预测须核对覆盖范围并执行统一评分/NMS。
  val2017 用于开发/调参，不能宣称为独立最终测试集。当前没有训练 A/B/C，也没有
  完成新的全量 AP/AR 基线评估。
- 原图叠加可视化验收仍可由负责验收的组员补充，数值联调不能代替所有视觉检查。
