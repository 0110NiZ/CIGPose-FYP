"""Basic settings for Module C training."""
from pathlib import Path

WORK = Path('/home/wanqiao/projects/cig-module-c')

TRAIN_CACHE = Path('/mnt/e/FYP/CIGPose_shared_cache_full_20261112/train2017')
VAL_CACHE = Path('/mnt/e/FYP/val_detection_cache/val2017-detection')

TRAIN_ANN = Path('data/coco/annotations/person_keypoints_train2017.json')
VAL_ANN = Path('data/coco/annotations/person_keypoints_val2017.json')

BASE_MODEL = Path('checkpoints/cigpose-m_coco_256x192.pth')

OUTPUT = WORK / 'runs/module_c_seed42'

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EVAL_EVERY = 5

SEED = 42
DEVICE = 'cuda:0'

LABELS = WORK / 'labels/train2017_c_labels.npz'
TRAIN_DATA = WORK / 'data/train2017_c_features.npz'
VAL_DATA = WORK / 'data/val2017_c_features.npz'
