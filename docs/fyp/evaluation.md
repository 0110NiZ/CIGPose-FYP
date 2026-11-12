# 统一 GT 框 / single-view AP/AR 评估协议

2026-11-12 已确定：主实验统一 **GT 框、single-view**。M 不需要重新选择框来源。

## 固定配置

正式入口：
`mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_coco-256x192_gt-single-view.py`

它继承原始 CIGPose-m 配置，仅明确覆盖评估设置。原始配置保留用于历史复现，
仍有 flip_test=True，不能误用作本次主实验入口。

| 项目 | 统一设置 |
|---|---|
| 基础模型/输入 | CIGPose-m，256×192（高×宽），float32 |
| checkpoint | checkpoints/cigpose-m_coco_256x192.pth；SHA-256 见 interfaces.md |
| 框来源 | 原始 COCO GT bbox；val/test 的 bbox_file 均为 None |
| 框分数 | 数据集提供的 GT bbox_score=1.0 |
| 翻转 | flip_test=False；不生成翻转视图，不融合预测 |
| 预处理/解码 | 沿用原配置确定性 val_pipeline、SimCC codec 和原图坐标恢复 |
| 姿态分数 | CocoMetric 的 bbox_keypoint：框分数 × 分数严格大于 0.2 的关键点平均分；无符合点则为 0 |
| 后处理 | oks_nms，阈值 0.9，执行一次 |
| NMS 实例面积 | 沿用 CocoMetric：取预处理后的 bbox_scales 两维乘积；缓存导出时对应 input_scale 两维乘积，不用 GT annotation area 替代 |
| 评估标注 | 完整原始 person_keypoints_val2017.json |
| 评估器 | 仓库现有 CocoMetric / COCO 标准 keypoints AP/AR |

原始 CIGPose、P0、固定 Top-1+alpha=1，以及 A/B/C 的所有比较使用同一配置。
所有模型推理使用 eval/no-grad；基础模型权重不更新。
结果必须标注 **COCO val2017 / GT boxes / single-view**，不能称为检测框成绩。
val2017 仍用于开发/调参，不是独立最终测试集。

## GPU 端运行原始 CIGPose 基线

从仓库的 `mmpose/` 目录运行，数据位于仓库根目录 `data/coco/`：

```bash
python tools/test.py projects/cigpose/body_2d_keypoint/cigpose-m_coco-256x192_gt-single-view.py ../checkpoints/cigpose-m_coco_256x192.pth --work-dir ../outputs/eval/original_gt_single_view --cfg-options test_evaluator.outfile_prefix=../outputs/eval/original_gt_single_view/predictions
```

使用有 PyTorch/MMCV/MMPose 和 GPU 的环境。数据/权重路径可按机器调整，内容及哈希
保持一致；保存最终展开配置、实际命令、代码版本、日志、预测文件与指标。
这条命令运行的是原始干预逻辑；它不会自动产生 P0、A/B/C 等其他模式。
P0/强制干预/模块结果需要接入对应前向并导出，统一使用相同 codec、坐标恢复、
评分与 NMS。原始逻辑与固定 Top-1+alpha=1 是否等价应先验证，再决定是否复用结果。

## M 的 CPU 评估入口（2E）

M 负责把导出的预测文件接到现有标准评估实现，不加载基础模型、不跑全量模型前向。
GPU 端由 N/T 配合生成少量真实样例及完整基线预测。

优先约定两种文件边界，文件说明中必须写清所处阶段：

1. **后处理前的实例预测**：image_id、annotation_id（供审计）、category_id=1、
   原图 keypoints[17,2]、解码 keypoint_scores[17]、bbox[4]（注明 xyxy）、bbox_score=1、
   预处理后的 bbox_scales/input_scale[2]（用于一致的 NMS 面积）。
   不要用初始 uncertainty/confidence 代替最终 keypoint_scores。
   若 CPU 端接收此格式，应复用相同评分和 OKS NMS，且使用相同预测面积定义。
2. **最终 COCO 预测 JSON**：image_id、category_id、keypoints（51 个数）和最终 score。
   由 CocoMetric 导出的该文件已评分并做 NMS；CPU 端只计算 COCO 指标，不再重复处理。

用 GPU 端的原始基线输出作为对照，让 CPU 重算结果与其 AP/AR 在输出精度内一致。
少量样例仅验证格式和流程，不作为正式成绩。若独立使用 COCOeval，必须保持
相同的 keypoints 参数、sigma、maxDets、面积范围及完整评估图片集合；不可只按
预测出现的 image_id 缩小评估范围。可先使用运行基线所用的 xtcocotools 实现，
使用其他 COCOeval 实现时先完成数值对照。

## 预测覆盖范围与现有缓存

完整评估器始终读取原始 val2017 标注；不要把缓存的 selected_annotations.json
作为新的评估真值文件。正式模型输入按上述配置的数据集过滤执行。
有效人体清单用来核验导出前的实例身份和覆盖范围；NMS 之后的预测数可以减少，
不能要求最终 COCO JSON 必须恰好含 6,352 条。

现有 GT 框缓存与主实验的框来源及 flip 协议一致，无需重建。但缓存本身不是
AP/AR 成绩。若从缓存导出 P0/候选预测，必须先核对其完整人体集合、bbox/score、
坐标、精度和后处理与正式入口一致，再比较。GT keypoints/area/OKS/gain 只用于
离线监督和评估，不能用于模块决策或预测分数计算。

## 4B 与交付

- N/T：运行 GPU 推理，提供预测、实际配置、checkpoint 标识、commit、命令和日志。
- M：CPU 计算/复核各基线 AP/AR，记录同条件对照结果，识别不一致或缺失项目。
- M 交付：CPU 评估脚本、依赖/格式/命令说明、结果表、日志及可追溯元数据。
- 历史 AP 0.766/AR 0.793 只有经日志确认全部条件一致才能作为同条件成绩；
  不能因基础权重相同就直接复用。现阶段未重跑新的全量 AP/AR 基线。
