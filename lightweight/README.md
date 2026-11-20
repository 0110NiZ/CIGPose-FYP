# CIGPose 当前轻量化实验（统一batch128）

只维护当前团队实验：小骨干、256维Head、30%固定分层训练子集、FP16、训练batch128 / 验证batch64、100轮。
原版CIM baseline已经完成，最佳AP **70.2148**（第100轮，完整val2017、GT框、开启翻转测试）。

- [统一超参数、四人CIM分工与交付规则](TEAM128.md)
- [当前唯一训练配置](configs/team128_baseline100.py)
- [基线结果、loss与AP曲线](results/README.md)

## 文件与用途

| 文件 | 用途 |
| --- | --- |
| configs/team128_baseline100.py | 所有人继承的统一配置 |
| data/selected_image_ids.json | 冻结的16983张训练图片ID |
| data/manifest.json | 分层规则、44941个人体实例、原始及子集标注摘要 |
| scripts/prepare_subset.py | 直接按固定ID重建当前子集，组员使用这个 |
| scripts/sample_stratified30.py | 原始分层抽样算法的可移植版本，仅用于在新目录审计复现 |
| scripts/train_team128.py | 统一训练入口，支持20轮初筛及恢复到100轮 |
| results/ | 本次100轮基线结果与曲线 |

## 环境和数据

先激活CIGPose环境，再从仓库根目录安装当前代码：

```bash
python -m pip install -e ./mmpose --no-deps
```

已验证5090环境：Python3.12、PyTorch2.8.0+cu128、CUDA12.8、MMCV2.1.0（为sm120编译）、MMEngine0.10.7、MMDetection3.2.0、NumPy1.26.4、Albumentations1.3.1。
4060已有环境为Python3.8、PyTorch2.0.1+cu118；环境差异与对照规则见TEAM128.md。其余安装要求见原项目文档，不应给5090使用不支持其架构的MMCV包。

官方COCO图片及标注放在或链接到仓库根目录`data/coco/`：train2017、val2017、annotations；保留原配置需要的person_detection_results文件。不需要test2017。不要修改原始图片或标注。

```bash
python lightweight/scripts/prepare_subset.py
```

脚本验证原始标注摘要、训练验证互斥、图片/实例数量和最终子集摘要。应生成16983张图片、44941个人体实例，SHA256：
`ec3ff51924cf1be81e73e7b175f33a380b913fa51c1de5b9bd64a235d491aa5a`。
图片内容不逐张校验，请使用未经改动的官方图片。所有人复用固定名单，不能自行重新抽样。

如需审计分层算法，可在已配置数据后执行（必须指定尚不存在的目录，不覆盖固定名单）：

```bash
python lightweight/scripts/sample_stratified30.py --output work_dirs/sampling_audit
```

## 开始训练

统一配置会从官方OpenMMLab下载小骨干预训练权重。以下命令从头训练原版CIM基线：

```bash
mkdir -p work_dirs
nohup python -u lightweight/scripts/train_team128.py \
  --work-dir work_dirs/team128_baseline \
  > work_dirs/team128_baseline_console.log 2>&1 < /dev/null &
tail -f work_dirs/team128_baseline_console.log
```

首次训练要求新的结果目录。重复执行命令不能解决训练问题；runner仅对相同work-dir上锁，不阻止其他目录占用同一GPU。Ctrl+C退出tail不会停止后台训练。
四人各自使用独立配置和结果目录，20轮筛选及续训命令详见TEAM128.md。

数据集图片、生成标注、权重、密码及完整日志不上传Git。原版模型实现及依赖文件保留；团队只使用本目录中的统一配置。
