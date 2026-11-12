# 真实 P0/Pc 小样本联调清单（待执行）

此文件是与 TANG/NIE/MAI 的交接要求，不是已完成的实验记录。

## 输入和身份

先选固定 internal-dev 中 3–5 个有效人体，覆盖有遮挡和清晰标注。用 image_id + annotation_id 定位，不靠 DataLoader 行号匹配。检测框正式评估使用另定的实例匹配协议，不能强行把 det_idx 当作 annotation_id。

每个实例保存：image_id、annotation_id、split 名称、split_hash_method、split_sha256、权重/配置/commit、原图尺寸、原始 GT bbox、GT keypoints、GT area、裁剪与坐标还原信息、关节编号、alpha、P0/Pc 的原图坐标及分数、两者 OKS 和 gain。

## 模型路径检查

- P0：不替换特征，仍经过同一 GNN 和解码。
- Pc：从同一份未干预状态独立生成，不在前一个候选上继续修改。
- 基础模型参数和 BatchNorm 状态不变；明确框、翻转、解码和评分设置。
- alpha=0 恢复 P0，gain 在约定数值容差内为 0；alpha=1 与同关节完整替换一致。

## 坐标和 GT 检查

由 TANG 导出真实模型结果与还原过程。确认网络输出经过 codec 解码及正确的原图映射，不能只因为数值落在图片范围内就认定坐标正确。

将 P0、Pc 与同一人体 GT 叠加可视化，记录检查者。两者统一使用该 annotation 的 area 和 v>0 标注。逐项调用公共 compute_instance_oks 和官方 COCOeval.computeOks 对照；计算 gain 并保存原始坐标。

这些配对数值是干预监督，不等同于正式 AP/AR 评价。GT 和真实 gain 不得进入实际决策输入。

## 真实数据读取检查

由实际数据生成/训练入口导出所读取的 image_id 和 annotation_id：

1. train 样本的 image_id 属于固定 train，annotation_id 属于 train_valid_ann_ids。
2. dev 样本对应固定 dev 及 dev_valid_ann_ids。
3. 同一个 annotation_id 的全部关节、强度候选保持同一划分。
4. 检查 train/dev 的实际图片 ID 无交集；不得运行时重新随机切分。
5. 记录过滤前后数量；小样本不能证明完整数据加载无泄露，扩大运行时继续保留这些断言/检查。

## 验收与下一步

保留 3–5 个真实样例的坐标、可视化、身份检查、官方 OKS 对照和实际加载 ID 记录，由相关组员复核。模型输出接口和训练读取代码未提供，当前不能自动补做这些结果。小样本链路通过后，再生成小规模收益缓存，最后扩大；暂不生成全量收益数据。
