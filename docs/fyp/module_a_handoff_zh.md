# A 模块：N/Y 当前交接协议

更新：2026-11-16。Y 已提供原始全量标签；N 完成诊断，在用户批准的“旧 TF32 精度 + 全局确定性”下重生成独立 V2 标签并通过验收。**不再等待标签；尚未训练，没有 A 的 AP 提升结论。**

## 正式版本

使用 train2017_legacy_deterministic_v2_20261116：4,682 分片、149,813 人、2,546,821 个 Gain。Y 原包和严格旧开关下未通过复验的 V1 均保留，不混用。

遵循[公共精度协议](precision_policy.md)。详细设计、训练验证和路径见[完整方案](project_status_20261116.md)及[接口说明](module_a.md)。

## 标签接口

`gain[j] = OKS(Pj, GT) - OKS(原缓存P0, GT)`，COCO 关节顺序 0..16，α=1。候选从原特征独立生成，坐标统一到原图，使用共享 OKS 的 GT 面积和可见性规则，保留负收益。

每个 NPZ 仅包含 image_id[N]、annotation_id[N]（int64），oks_before[N]、oks_after[N,17]、gain[N,17]（float32/float64），禁止 pickle。按 annotation_id 对齐并核对 image_id，不按行号拼接；全部人体恰好一次。公式校验绝对容差 2e-6。

metadata 需要完整数量、分片哈希和真实来源。来源可用真实生成提交，或经过读取器校验的源码快照及 SHA256；不能虚构 Git 提交。本次 V2 使用源码快照。旧 Y_metadata_template.json 是不完整模板，不是正式 metadata。

全量检查和独立 GPU 抽查 51,680 候选通过，最大 OKS 差约 4.44e-16，最佳关节变化 0。最终状态依据 acceptance_report.json、metadata.json、reader_check.json；generation_summary.json 的 accepted:false 是生成阶段状态。

重算 P0 仍有少量差异，不声称与历史缓存完全一致；正式标签基准始终是原缓存 P0，此定义也必须同步给 B/C。

## 分工

| 人员 | 后续工作 |
|---|---|
| N | V2 生成验收和接口已完成；获明确训练指令后训练 MLP、完整 val 评估 |
| Y | 配合标签审计、选择策略对照、离线分析和可视化 |
| B/C 负责人 | 同步公共精度初始化、候选接口和 P0 定义 |

只训练 MLP。A-only 固定 α=1、始终接受，全负也选一个，回退属于 C。完整 val 的 104,125 条检测进入统一后处理，AP/AR 使用全部 5,000 张图，不用 GT 预删误检。

训练入口默认 preflight，显式 --run-training 才更新参数。本次未运行该开关。代码和小报告通过 Git 交付，标签 ZIP、缓存和 checkpoint 单独交付。
