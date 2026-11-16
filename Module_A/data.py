"""读取数据：复用已经验收的数据加载器，避免两套标签匹配逻辑。"""
import common
from a_dataset import ATrainingData, fit_input_statistics

# ATrainingData(train_cache, labels) 先检查完整性和人体 ID。
# data.batches(...) 每次给出 inputs、17 个 gain 标签、人体 ID。
# inputs 只有 features、uncertainty、pose_coords、pose_scores。
# fit_input_statistics 只用 train 计算标准化参数，不读取 val 统计量。
