# 统一 CPU AP/AR 评估说明与 4B 基线核验记录

本文档对应任务 **2E（统一 CPU 评估入口）** 与 **4B（全量基线核验）**，严格遵循 `docs/fyp/evaluation.md` 协议。

## 1. 脚本位置与环境依赖

- **脚本路径**：`mmpose/analysis/fyp/eval_unified.py`
- **分支与提交**：`mai/eval-and-module-c`（commit `06b5b42`）
- **运行环境**：纯 CPU 即可，无需加载 CIGPose 模型、无需 `mmcv` 或 GPU。
- **核心依赖**：
  - Python >= 3.8
  - `numpy`
  - `torch`（CPU 版即可）
  - `pycocotools`

## 2. 支持的输入阶段与格式

通过 `--stage` 参数指定（默认 `auto` 自动识别）：

1. **`coco_json`（已后处理 COCO 预测 JSON）**
   - 格式：`list[dict]`，每条包含 `image_id`、`category_id`、`keypoints`（51 个数值）、`score`。
   - 处理方式：**跳过二次打分和 OKS NMS**，直接在完整原始 `person_keypoints_val2017.json`（5,000 张图片）上调用 `COCOeval`。
2. **`pre_nms`（后处理前实例预测）**
   - 格式：包含 `image_id`、`keypoints`/`coords` (`[N, 17, 2]`，原图绝对坐标)、`keypoint_scores`/`scores` (`[N, 17]`)、`bbox` (`[N, 4]`)、`bbox_score`、`bbox_scales`/`input_scale` (`[N, 2]`)，可选 `annotation_id`。
   - 处理方式：复现 `CocoMetric` 规则执行 `bbox_keypoint` 打分（`keypoint_score_thr=0.2`）与 `oks_nms`（`nms_thr=0.9`，实例面积取 `bbox_scales` 两维乘积并在匹配时取两实例平均值），再运行 `COCOeval` 并输出覆盖率与重复 `annotation_id` 审计。

## 3. 运行命令示例

```bash
python mmpose/analysis/fyp/eval_unified.py \
  --ann-file <path_to/person_keypoints_val2017.json> \
  --pred-file <path_to/predictions.keypoints.json> \
  --stage auto \
  --exp-name <experiment_name>
```

评估结果与可复现元数据（`git_commit`、`flip_protocol`、`audit`、10 项指标）默认保存至 `experiments/baseline/<exp_name>_eval.json`。

## 4. 4B 全量基线核验结果

**统一协议**：`COCO val2017 / GT boxes / single-view`

- 配置：`mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_coco-256x192_gt-single-view.py`
- 权重：`checkpoints/cigpose-m_coco_256x192.pth`（SHA-256: `3dd69c8ab4e3c7a923a0183fd9abf844cb7ef13d39d26693baa082ca7931b3d0`）
- 评估条件：`flip_test=False`，GT 框，`bbox_keypoint`（thr=0.2），`oks_nms`（thr=0.9），全量 5,000 张 `val2017` 图片。

| 基线模式 | 推理人数 | NMS 后预测数 | AP | AP50 | AP75 | AR | 报告文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| Original CIGPose | 6,352 | 6,336 | 0.75481 | 0.92565 | 0.82687 | 0.78292 | `experiments/baseline/baseline_original_gt_single_view_eval.json` |
| Non-Intervention (P0) | 6,352 | 6,337 | 0.75574 | 0.92578 | 0.82709 | 0.78350 | `experiments/baseline/baseline_p0_gt_single_view_eval.json` |
| Fixed Top-1 + alpha=1 | 6,352 | 6,336 | 0.75481 | 0.92565 | 0.82687 | 0.78292 | `experiments/baseline/baseline_top1_alpha1_gt_single_view_eval.json` |

### 核查说明

1. 三组 `coco_json` 重算结果与 GPU 端 30 项参考指标完全一致（最大差值 `0.0`）。
2. `pre_nms` 路径在修正 OKS NMS 两实例面积平均值公式后，于 `original/raw_predictions_before_nms.json` 实测通过（NMS 后 6,336 条，`AP=0.75481 / AR=0.78292`）。
3. 历史 `AP 0.766 / AR 0.793` 是在 `flip_test=True` 条件下得到，不混入本次 `single-view` 主实验基线；`val2017` 用于开发与调参，不作为独立最终测试集。