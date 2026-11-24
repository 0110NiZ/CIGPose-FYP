# COCO 50% 固定训练子集

保留原 30% 的全部 16,983 张图片和 44,941 个人体实例，按原有遮挡、人体大小、人数分层补抽。

最终 28,305 张图片、74,907 个有效人体实例，人体比例 50.000334%。

训练时将子集标注放到 data/coco/annotations/，把训练数据集 ann_file 指向此文件。图片仍读取官方 COCO train2017/。验证继续读取完整 val2017 及官方验证标注。

selected_image_ids.json 是所有入选图片 ID，preserved30_image_ids.json 是保留名单，added_image_ids.json 是补入名单。manifest.json 记录逐层数量和原始/子集 SHA256。SHA256SUMS.txt 可用于核对子集包内容。

本包提供固定名单、标注、复现脚本和校验信息，组员配合官方 COCO 图片使用。

复现脚本需要 Python 和 NumPy，执行示例为

```bash
python extend_coco30_to50.py --train person_keypoints_train2017.json --val person_keypoints_val2017.json --ids30 original30/selected_image_ids.json --manifest30 original30/manifest.json --annotations30 original30/person_keypoints_train2017_stratified30_seed21.json --out new50
```
