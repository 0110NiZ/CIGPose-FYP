# T 分支审查：6fd5b3c

范围：静态阅读 interfaces.md、cig_head.py 的新增代码及两个 smoke scripts。
初次静态审查时未接入模型环境。2026-11-12 后续已在本机 WSL 的 cigpose 环境完成
16 人缓存/真实 OKS 联调及全量缓存生成；验证范围与报告见 shared_cache_handoff.md。
这不等于已逐项重跑 T 的原始两份 smoke scripts。

## 可以直接复用

- extract_refinement_state 复用原 GAU 特征路径，P0 明确绕过 CIM、仍经过 GNN。
- apply_intervention 克隆输入，只修改对应 batch 的指定关节，支持逐实例 alpha。
- 下游统一为 predict_from_joint_features，既有普通 forward 未被改动。
- apply_intervention 的真实位置是 head.cim；extract_refinement_state 返回的是
  p0_simcc，不是已还原的原图坐标。2C 已按实际 API 接入。

## 本次处理的事项

1. interfaces.md 1.2 仍是旧 train2017 90/10 internal-dev 协议。根据当前用户明确要求，
   已记录改为完整 train2017 训练、val2017 开发的协议修订，并保留历史解释。
2. 头部接口本身不保证整个模型 eval/frozen 或外层 flip 设置；它们是调用者责任。
   2C 显式关闭 flip、冻结参数、eval、inference_mode，并使用确定性 pipeline。
3. check_topdown_alignment.py 缺少图片 ID 时 image_id_ok=True；annotation ID 仅从
   dataset.get_data_info(b) 打印，没有严格与模型 sample 身份交叉核验。这不能单独
   证明完整人体对齐。2C 缺 ID 直接报错，逐人体对照原始标注和有效名单，并检查数据集
   最终载入的完整 ID 映射与预期相等，不靠 loader 行号。
4. 该 smoke 的坐标比较是在检验还原公式与 add_pred_to_datasample 的一致性，
   不能替代与原图/GT 的独立可视化核验。Y 的真实样例验收仍需执行。

## 后续可加强（未改 T 的模型代码）

- apply_intervention 在校验前将 joint_ids 转成 long，非整数输入可能被截断。
  调用者必须传整数关节 ID，后续可在接口内增加原始类型/整数性检查。
- 可补充 alpha=NaN/越界、错误 batch shape、空 batch 和设备/dtype 的测试。
- 公共接口和 T 的 smoke scripts 不提供完整训练/评估数据入口；2C 不声称已完成
  检测框 AP/AR benchmark 或全部监督标签生成。
