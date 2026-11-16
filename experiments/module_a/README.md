# A 模块检查记录

更新：2026-11-16，基于 public-tasks@66edb12，分支 feature/module-a。最新状态见 docs/fyp/project_status_20261116.md。

## 当前 V2

正式采用旧 TF32 精度 + 全局确定性。全量标签 4,682 分片、149,813 人、2,546,821 Gain；独立 GPU 复验 51,680 候选，最大 OKS 差 4.44e-16，最佳关节变化 0。

v2/ 保存本次全量预检查、真实接口检查和 32 项测试日志。读取检查覆盖 train 149,813、val 104,125；GPU 接口抽查 96 train + 93 val 人体。只用随机未训练 MLP 检查接口，未计算 A 的 AP/AR，没有优化器更新。

全量验收报告和读取报告也保存在 v2/。标签包和缓存单独交付，不上传 Git。

## 历史记录

本目录顶层原有 interface_check.json、test_report.json、各测试日志和 Y_metadata_template.json 是早期 N 端交接记录，原样保留，并非当前 V2 源码的哈希报告。元数据模板仍刻意 incomplete，不得用来替代正式标签 metadata。

Y 后来已提供标签；N 已重生成并验收正式 V2。旧报告中的“不生成标签”只描述当时的接口检查，不代表当前项目进展。

下一步训练需要用户明确指令；尚无收敛、收益预测或 AP 提升结论，test-dev 未运行。
