# val2017 固定检测框读取交付

## 已确认的读取协议

- 文件：COCO_val2017_detections_AP_H_56_person.json。
- 原文件 SHA-256：53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59。
- 原始记录 104125 条；score threshold=0.0，不额外筛选分数，不做额外 bbox NMS 或去重。
- 不根据 GT 人体清单筛选，不使用旧 90/10 split。
- 保留原始框、分数及顺序。无效记录直接报错，不静默删除。
- pose scoring、OKS-NMS/pose NMS 由 M 的评估入口处理，本脚本不做。
- 全程 single-view 是模型端的协议，由 T 的前向配置落实；本脚本不运行模型或翻转图片。
- 上游检测器确切 config/checkpoint、上游 NMS 不能从 JSON 推断，元数据保留为空或未提供；固定文件及 SHA 是当前 val 输入依据。不据此宣称 test-dev 检测来源已解决。

## 文件和字段

scripts/read_val_detection_boxes.py 是标准库入口，不依赖 torch 或 mmpose。

load_detection_instances(detection_path, annotation_path, image_root=None) 返回 metadata, instances。

每个实例含：dataset_split、image_id、detection_instance_id、instance_id、source_detection_index、category_id、bbox、bbox_format、bbox_xyxy、bbox_score、image_file_name、image_width、image_height。传入 image_root 时增加 image_path。

- bbox 为原图 XYWH，不裁剪、不扩框、不加 padding；bbox_xyxy 是明确命名的转换字段。
- bbox_score 保持源 score，不改成 1，也不与关键点置信度混淆。
- detection_instance_id = val2017:det:<完整源文件SHA256>:<原始零基行号，至少6位>。instance_id 为相同别名。
- ID 不依赖 DataLoader 顺序或样例导出顺序。更换或重写原检测文件会触发 SHA 检查失败，不悄悄复用 ID。
- 不提供 annotation_id。检测与 GT 的关联需另定规则；正式 COCO 评估按评估器协议匹配。
- source_annotation 只用于图片 ID、文件名及尺寸；不读取 annotations 进行人体筛选。

## 在 LIN 的 Windows CMD 运行

将 scripts、tests、docs 合并复制到 F:\FYP\CIGPose-FYP-git，不删除已有目录。

```cmd
cd /d F:\FYP\CIGPose-FYP-git
py -m unittest discover -s tests -p test_val_detection_reader.py -v
py scripts\read_val_detection_boxes.py --detections "F:\FYP\data\coco\person_detection_results\COCO_val2017_detections_AP_H_56_person.json" --ann "F:\FYP\data\coco\annotations\person_keypoints_val2017.json" --out-dir "experiments\val_detection_input_v1"
```

第一次不传 image-root，只生成真实检测记录和图片相对文件名，没有检查本机图片存在。生成数量应该为 104125 个检测实例、3893 张有检测的图片，ID 数 104125。

输出：

- val2017_detection_manifest.json：元数据及全部稳定实例 ID；原始框和分数仍从 SHA 固定的源 JSON 读取。
- val2017_detection_report.json：协议、数量和读取检查结果。
- val2017_detection_samples.json：默认首3张不同图片，各最多2条原始记录。仅为联调子集，不改变完整输入。

找到本机 val2017 JPG 目录后，增加 --image-root 参数并换一个输出目录，例如 --out-dir experiments\val_detection_input_images_v1。脚本会检查导出的样例图片文件是否存在，但不解码图片或检查全部图片，也不完成模型前向。

输出目录已有同名文件时拒绝覆盖。重新检查请使用新的输出目录。

## TANG 读取示例

在仓库根目录执行 Python，示意：

```python
import sys
sys.path.insert(0, "scripts")
from read_val_detection_boxes import load_detection_instances

meta, instances = load_detection_instances(
    "/path/to/COCO_val2017_detections_AP_H_56_person.json",
    "/path/to/person_keypoints_val2017.json",
    image_root="/path/to/val2017",
)
sample = instances[0]
# sample['bbox'] 是 XYWH；若模型接口要 XYXY，明确使用 sample['bbox_xyxy']。
# 由模型端继续计算 input_center/scale 和 crop，保留原 bbox_score 与 ID。
```

T/N 应保留 metadata 的 SHA、协议及实例 ID；不得再次执行隐含 score threshold，不能在 MMPose 默认筛选或 NMS 中丢框后仍声称输入是全量104125条。

## 验收范围

本包带8项合成输入回归测试，验证保留零分、重叠框、不依赖GT人体、ID稳定、坐标转换、异常报错、输出保护和样例图片路径检查。合成测试不是模型实验。

下载源检测文件未附在本包中；完整104125条真实记录须在用户本机执行上述命令核验。随后检查样例图片、将样例交给TANG，再由T/Y验证真实前向、坐标恢复、alpha=0/1。此处不宣称这些模型检查已通过。
