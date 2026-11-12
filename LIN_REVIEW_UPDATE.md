# LIN 划分与 OKS 复核修改说明

本包只修改可复跑性与校验方式，不重新划分图片、不改人体过滤规则、不改 OKS 公式。

## 已完成

- 新增 scripts/fyp_data_common.py：统一 JSON 读取、校验、报告写入及原 v2 人体过滤。
- 校验方法：解析 JSON 后，sort_keys=True、separators=(",", ":")、ensure_ascii=False、allow_nan=False，编码 UTF-8，再计算 SHA-256。换行、缩进、对象键顺序和 UTF-8 BOM 不影响结果；图片 ID、数组顺序及元数据仍影响结果。
- 新字段 split_hash_method 明确区分新旧算法，不会默默接受错误的旧校验值。
- 原始 train_dev_split.json 按字节保留，图片划分完全不变。
- manifest、数据报告、OKS 报告的校验字段已迁移，并记录旧值及 results_recomputed=false；已有统计和数值结果没有重算或改写。
- 新 canonical split SHA-256：a1eaaef41a41c50ab9ad1cc4fcf35a8ef898afce7fbec385ce14c2ccdb4f5351。
- 原 raw-file SHA-256：99b3f6283abecab254228cf92b8bdf5f05b2dcec60082822c69b98ba7538f214。
- 三个主脚本均支持运行时标注路径和输出目录；OKS 检查还支持自选 split 与 manifest。
- 数据/OKS 检查默认拒绝覆盖已有结果，可显式使用 --overwrite；生成新 split 没有覆盖开关。

## Windows 本地应用步骤

1. 保持当前 .venv-oks 环境。将修订 ZIP 下载到 F:\FYP，解压到独立文件夹（例如 F:\FYP\lin_review_fixed）。
2. 将其中的 scripts、splits、mmpose、experiments、tests、docs 目录和本说明合并复制到 F:\FYP\CIGPose-FYP-git。选择替换本包中的同名文件；保留仓库里其他文件。不要删除原目录。
3. 修改的是本包列出的文件；不包含图片、原始 COCO 标注或虚拟环境。无需重新运行 build_train_dev_split.py。
4. 在仓库根目录运行：

```cmd
python -m unittest discover -s tests -p test_lin_portability.py -v
```

5. 使用本机完整标注重新检查到独立目录：

```cmd
python scripts\check_split_data.py --ann "F:\FYP\data\coco\annotations\person_keypoints_train2017.json" --split "splits\train_dev_split.json" --out-dir "experiments\review_recheck\data"
```

6. 使用刚重建的 manifest 做官方 OKS 对照：

```cmd
python mmpose\analysis\fyp\check_oks.py --ann "F:\FYP\data\coco\annotations\person_keypoints_train2017.json" --split "splits\train_dev_split.json" --manifest "experiments\review_recheck\data\train_dev_valid_ann_ids_v2.json" --out-dir "experiments\review_recheck\oks"
```

两次独立复跑输出不会覆盖 splits 下的已提交文件。再次使用同一输出目录时，需增加 --overwrite 或选择另一个目录。复跑后将新报告与已提交数据报告比较，确认统计一致。所有相对路径参数以当前工作目录为基准；默认路径以脚本所在仓库为基准。

## Linux 或其他机器复跑

```bash
python -m pip install numpy pycocotools
python scripts/check_split_data.py --ann /path/to/person_keypoints_train2017.json --split splits/train_dev_split.json --out-dir /tmp/lin-check/data
python mmpose/analysis/fyp/check_oks.py --ann /path/to/person_keypoints_train2017.json --split splits/train_dev_split.json --manifest /tmp/lin-check/data/train_dev_valid_ann_ids_v2.json --out-dir /tmp/lin-check/oks
```

## 新划分生成器（已有团队 split 不需要运行）

```bash
python scripts/build_train_dev_split.py --ann /path/to/person_keypoints_train2017.json --out-dir /tmp/new-split --seed 42 --train-ratio 0.9
```

生成器现在与检查器共用完整 v2 过滤逻辑，并输出 split、v2 数据报告及有效人体清单。不会覆盖已有团队 split。不再把早期简化过滤的 train_dev_split_report.json 当作有效人体的权威统计。

## 仅迁移已有旧报告（本包已经执行，无需再做）

```bash
python scripts/migrate_split_checksums.py --out-dir /tmp/migrated
```

该命令先核验旧校验值对应输入 split 的 LF/CRLF 字节形式，再迁移报告元数据。旧值无法核验时拒绝迁移，不直接把不匹配报告改成“通过”。本命令不读取完整标注、不重新计算旧报告结果、不修改 split。

## 验证证据和限制

- tests/test_lin_portability.py 共 8 项回归测试已在 Linux 通过，模拟 CRLF/LF、不同格式、BOM；没有宣称已在原生 Windows 运行。
- 在合成标注夹具上重新执行官方 COCOeval 的 19 组数值对照，最大绝对差为 0。
- 与原版过滤函数进行夹具对照，一致；原始 split 字节、完整有效人体 ID、原统计及原 19 组 OKS 数值均保持不变。
- experiments/oks/portability_revision_validation.json 记录本轮验证范围。
- 本次上传包不含完整 COCO 标注，因此此处没有重新执行 118,287 张图片的全量数据统计，也没有重跑原 5 个真实 GT 样例。原始真实 GT 数值报告仅作校验元数据迁移，需在本机按上面命令复跑。
- 真实 CIGPose P0/Pc、坐标还原、人体对应和训练数据读取验证尚未完成，见 docs/fyp/real_output_validation.md。

## 提交 GitHub

先本机复跑并检查 git status、git diff。继续使用 lin 分支，不需要重新建分支或重新划分。只暂存本包修改及确认过的新复跑报告，不使用 git add . 上传原始数据或环境。不要把本轮数值核验表述为真实模型实验完成。
