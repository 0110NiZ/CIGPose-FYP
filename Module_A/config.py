
"""Module A / A2: Global Context + OKS Gain Ranking configuration."""
from pathlib import Path

WORK = Path('/home/nzd/projects/cig-module-a-n')
TRAIN_CACHE = Path('/home/nzd/projects/cig-shared-cache-run-20261112/train2017')
LABELS = WORK / 'labels/train2017_legacy_deterministic_v2_20261116'
VAL_CACHE = Path('/home/nzd/projects/cig-detection-public-20261116/val2017-detection')
VAL_ANN = Path('/home/nzd/projects/CIGPose/data/coco/annotations/person_keypoints_val2017.json')
BASE_MODEL = Path('/home/nzd/projects/CIGPose/checkpoints/cigpose-m_coco_256x192.pth')

# A2 must not overwrite any previous A0/A1 run.
OUTPUT = WORK / 'runs/module_a_a2_ranking_seed42_bs1024_eval1'

# Keep the previously trained A1 Global Context architecture unchanged.
MODEL_ARCHITECTURE = 'a1_global'

EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EVAL_EVERY = 1
SEED = 42
DEVICE = 'cuda:0'

# A2 = original SmoothL1 + top-1 pairwise ranking loss.
GAIN_SCALE = 100.0
RANK_WEIGHT = 0.2
# In scaled Gain units: 0.05 / 100 = 0.0005 true OKS Gain.
# Nearly-tied candidates are excluded from ranking supervision.
RANK_MIN_GAP = 0.05
