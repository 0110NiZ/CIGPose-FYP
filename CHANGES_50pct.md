# 50% 对照实验改动清单

1. train.py：默认加载随包原样附带的固定 50% JSON；支持 --train-ann、--coco-root、--check 和 --diagnostic-cache；温度固定 0.1，检查 seed=21、batch=128、100 轮。输出目录独立为 group_e_adaptive_50pct_t01。
2. subset50.py（新增）：验证固定 JSON 的 SHA256；核对实际 dataset 的文件、ID 和 GT；按实际 dataset/sampler/dataloader 核验人数和 batch 数。记录版本、哈希、过滤数量、计划迭代数。拒绝重复抽样、多 GPU、丢尾批等改变原实验方式的配置。
3. tracking.py：删除训练 44,941 人/352 batch 断言，用真实 loader 结果替换；增加每轮及累计迭代数和实际优化器更新数。累计计数保存进 checkpoint 并在续训恢复。保留原所有损失日志口径。
4. 可选复用 30% 的五批诊断缓存：来源和样本核验通过后逐字节复制；缺少缓存或样本不在实际 dataset 时记录原因并沿用原生成流程；来源损坏或配置/GT 不一致时报错。
5. run_manifest.json：记录实际标注版本/哈希、loader 统计、优化器/调度/验证配置、诊断来源。续训验证实际 loader 与首次一致。完整验证标注按固定子集来源哈希核验。
6. 保留原 --resume 及本组 checkpoint 路径限制；新启动拒绝已有输出目录。图标题注明 50%。

没有改动 model.py、gradients.py；没有改动门控、动态调权、退火、损失函数、优化器输入、初始化或增强调度。原仓库外部配置不在本包内，继续原样继承。原验证人数 6,352 校验保留。

附带 CHANGES.patch、DELIVERY_VERIFICATION.json、16 项 CPU 测试与固定子集原始元数据。完整运行指令见 lightweight/group_e_adaptive/README.md。
