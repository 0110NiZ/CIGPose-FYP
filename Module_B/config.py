"""Settings shared by Module B. Paths below are for the existing WSL setup."""
from pathlib import Path

ALPHAS = [0.25, 0.5, 0.75, 1.0]
FEATURE_SPEC = {'name': 'b_source_features_v1', 'dimension': 563, 'fields': ['selected_joint_feature[256]', 'mean_joint_feature[256]', 'uncertainty[17]', 'confidence[17]', 'selected_joint_one_hot[17]'], 'excludes': ['GT', 'gain', 'candidate_predictions', 'oracle_alpha']}
STRENGTHS = tuple(ALPHAS)
WEIGHT_SHA = '3dd69c8ab4e3c7a923a0183fd9abf844cb7ef13d39d26693baa082ca7931b3d0'
EXPECTED_DETECTIONS = 104125
FIXED_BOX_SOURCE = 'fixed_precomputed_COCO_val2017_detections_AP_H_56_person'
GAIN_SCALE = 1000.0
NETWORK = '563-128-ReLU-Dropout(0.1)-64-ReLU-4'
ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/mnt/e/fyp_module_b')
TRAIN_CACHE = Path('/mnt/e/CIGPose_shared_cache_full_20261112/train2017')
DETECTION_CACHE = Path('/mnt/e/val_detection_cache/val2017-detection')
LABELS = WORK / 'full_train_v4'
TRAINING = WORK / 'train_b_full_v1'
EVALUATION = WORK / 'eval_b_full_v1'
ORIGINAL_TRAIN_SHA = '1592ed3cfa972a7a535f8225ac832cb3d83ccd4e9dcd4e5db197014e5ac9c359'
