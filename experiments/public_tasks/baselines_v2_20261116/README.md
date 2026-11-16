# 公共基线 V2 重跑

日期：2026-11-16。这里只包含公共基线；A 模块结果和比较放在 feature/module-a 分支。

使用旧 TF32 卷积精度 + 全局确定性 V2、固定 checkpoint、single-view、flip_test=False。完整 val2017 5,000 图、104,125 检测，不做 GT 预筛选。AP/AR 下表采用 0 到 100 标度；JSON/CSV 同时保留精确原值。

| 方法 | AP | AR |
|---|---:|---:|
| 原始 CIGPose | 73.5402 | 78.9704 |
| 不干预 P0 | 73.6223 | 78.9767 |
| 不确定性 Top-1，alpha=1 | 73.5402 | 78.9704 |

## 如何重跑

- P0 保留原缓存固定坐标和分数，重新执行评分、NMS、COCO 评估，不重算 P0 来替换基准。
- Top-1 使用原缓存特征和不确定性 Top-1，在 V2 设置下重新生成候选，alpha=1，始终接受。
- Original 从原图、相同检测框和同一批次顺序运行原始 CIGPose 前向。
- 三组分别通过官方 CocoMetric、公共 CPU pre-NMS 和最终 COCO JSON 三条评估路径；全部十项指标在 1e-12 容差内一致，后处理后的逐条预测也核对一致。
- 全部原始缓存、权重和标签保持原样。此次没有训练，没有优化器更新。

## 文件与复现

运行入口见 docs/fyp/evaluation.md 的 V2 重跑命令。完整预测在本机：

`/home/nzd/projects/cig-module-a-n/baselines/v2_20261116`

本目录仅放小型指标、日志、结果表和来源记录。每组 metrics.json 保存原始/最终预测文件的 SHA256，provenance.json 保存精度、数据和源码哈希。executed_runner.py 是本次实际执行源码快照，与 provenance 中哈希一致；运行脚本的后续日志修复不改动计算。

MMEngine 的部分日志输出曾因旧 stdout 已关闭出现日志警告；没有中断推理或评分，三组完整性和三条评估路径检查均通过。新版入口已把日志初始化放到重定向之前，避免该提示。原日志保留供追溯。

历史实验报告仍保留，本报告是当前 V2 的公共对照依据。不能由三组整体 AP 接近或相同推断逐人预测完全相同，详见各组 comparison_to_old_cache。
