"""平时改路径和训练参数，主要看这个文件。"""
from pathlib import Path

WORK = Path('/home/nzd/projects/cig-module-a-n')
TRAIN_CACHE = Path('/home/nzd/projects/cig-shared-cache-run-20261112/train2017')
LABELS = WORK / 'labels/train2017_legacy_deterministic_v2_20261116'
VAL_CACHE = Path('/home/nzd/projects/cig-detection-public-20261116/val2017-detection')
VAL_ANN = Path('/home/nzd/projects/CIGPose/data/coco/annotations/person_keypoints_val2017.json')
BASE_MODEL = Path('/home/nzd/projects/CIGPose/checkpoints/cigpose-m_coco_256x192.pth')
OUTPUT = WORK / 'runs/module_a_mlp_seed42'

# 第一轮先用这些参数；这是起始方案，还没有训练结果。
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EVAL_EVERY = 5
SEED = 42
DEVICE = 'cuda:0'
