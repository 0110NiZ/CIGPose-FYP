# 固定 50% 数据对照实验

以本次上传的 CIGPose_E_Adaptive_KD.zip 为修改基准。这是覆盖到原 CIGPose-FYP 仓库根目录的增量代码包，仍需原仓库中的 group_e_kd、group_c_kd、configs 和 mmpose，以及队友相同的 Linux 单 GPU 训练环境、COCO 图片和教师权重。

## 数据身份

包内 fixed_subset/ 是此前上传的固定 50% 标注包的原样副本；没有执行重新抽样。默认训练标注直接指向这里，因此不需要手工搬动标注。

- 版本：COCO50_nested30_fixed_annotations_seed21
- 标注 SHA256：bce0685bc39a2086881b4c434e931637371b68d06c2502438d271873d5a90f46
- JSON：28,305 张图片、74,907 条人体标注；保留 16,983 张原 30% 图片。
- 若实际 dataset 使用全部 74,907 条且不丢尾批，则为 586 batch/epoch，100 轮为 58,600 次训练迭代。实际值由运行时 dataset/sampler/dataloader 核验；AMP 可能使实际优化器更新数更少。
- 包内没有 COCO JPG、教师 checkpoint、学生预训练权重或队友的诊断缓存。

## 安装及命令

在与队友相同版本的仓库根目录解压。若此前已有 group_e_adaptive，请先备份该目录；不要覆盖正在运行实验使用的代码。

```bash
cd /root/autodl-tmp/CIGPose-FYP-50pct
unzip /root/autodl-tmp/CIGPose_E_Adaptive_KD_50pct.zip
```

以下可直接复制，沿用原仓库配置的图片位置和 DEFAULT_TEACHER。若这些默认路径尚未在你的机器上配置好，请在每条命令上加 `--teacher /实际的队友同款教师权重.pth` 和/或 `--coco-root /实际的COCO根目录`。路径需指向真实文件；不要使用不同教师权重来替代。COCO 根目录内部子路径仍按原配置的 data_prefix 与 val ann_file 解释。

先检查，不训练、不创建本组正式输出目录：

```bash
python -u lightweight/group_e_adaptive/train.py --check
```

独立启动，默认输出为 work_dirs/group_e_adaptive_50pct_t01：

```bash
python -u lightweight/group_e_adaptive/train.py
```

断点续训（与启动保持同样的路径参数）：

```bash
python -u lightweight/group_e_adaptive/train.py --resume
```

也可用 `--work-dir work_dirs/你的独立目录`，启动和续训必须相同。新跑拒绝已有目录；续训只接受本目录 last_checkpoint 指向的 epoch_N.pth，不接受其他组 checkpoint。保留 run_manifest.json 和 diagnostic_batches.pth。原来的每 10 轮保存策略不变，中断后未保存轮次会重跑；重启后的训练增强随机序列不保证逐位相同，与原版限制一致。

## 复用队友的五批增强输入

拿到队友同一次 30% Adaptive 实验的 diagnostic_batches.pth 和 run_manifest.json，把两者放在同一目录。首次启动时，例如：

```bash
python -u lightweight/group_e_adaptive/train.py --diagnostic-cache /root/autodl-tmp/diagnostics30/diagnostic_batches.pth
```

只提供可信队友生成的缓存。程序在真实 50% dataset 建立后核对缓存 SHA256、种子、温度、初始系数、学习率、batch、轮数、教师哈希、完整验证标注哈希、学生初始化、增强配方，以及未修改的模型/配置/梯度实现文件哈希。五批各 128 个样本必须在实际加载的 50% dataset 中，原始 GT 和 image_id 也要相同。

通过后逐字节复制缓存，直接使用已增强输入；不会根据 ID 再做一次增强。若样本不在实际 dataset 中，按原逻辑生成本组诊断批次并记下原因；若来源哈希、配方或 GT 不匹配则报错。没提供缓存时按原逻辑生成并明确标记，不能声称两组诊断使用相同增强输入。`--check` 检查数据与配置，缓存验证在正式启动的 before_train 中进行。

续训只加 --resume，不再传 --diagnostic-cache；始终读取本组保存的缓存和动态系数。

## 保持原有实验逻辑

model.py、gradients.py 与上传 ZIP 逐字节一致。遮挡门控、五批梯度诊断、前 80 轮动态调权、后 20 轮退火完全沿用。教师路径默认值、学生初始化、初始蒸馏系数、优化器、学习率调度、数据增强、EMA、完整验证和评估由原 E/C 配置继承。启动要求原配置为 seed=21、batch=128、100 轮，温度固定 0.1，不擅自改写不匹配的配置。

原验证损失人数 6,352 的校验保留；完整验证集没有改成训练子集。新增完整 val2017 标注哈希校验，必须符合固定子集 manifest 中记录的来源。

## 清单和日志

run_manifest.json 记录实际训练标注绝对路径、版本和 SHA256、实际 dataset 人数、sampler 人数、过滤数量、每轮 batch 数、100 轮计划迭代数、已加载 annotation ID 集合摘要、教师哈希、依赖源码哈希、实际配置及诊断缓存来源。

loss_history.csv 保留原字段和统计口径，新增 train_iterations、cumulative_train_iterations、cumulative_optimizer_steps；原 optimizer_steps 和 skipped_optimizer_steps 继续由实际 optimizer.step 回调计数。累计值随本组 checkpoint 恢复，不会从零重计。其他梯度日志、AP、summary 与原版一致，绘图标题注明 50%。

## 验证与边界

交付时核对固定数据包全部自带 SHA256、图片集合包含关系、标注 ID 唯一性，检查 Python 语法，执行 16 项 CPU 行为测试。可在包根目录运行 `python tests/test_subset50.py`（需要 PyTorch）。

当前环境未取得原仓库依赖配置，也不能运行你的 Linux CUDA/MMPose 环境，因此没有宣称已验证真实 dataloader、教师加载或 CUDA 前后向。请使用队友同一仓库版本与环境，先在训练机运行 --check。若实际数据加载结果与预期不同，应查看过滤数量和配置，不应硬改成人数 74,907 或批次数 586。启动时会核验实际使用的标注和 loader；精确的两组教师/配置一致性仍需队友的原配置或运行清单佐证。
