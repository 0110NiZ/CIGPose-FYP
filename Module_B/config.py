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
TRAINING = WORK / 'train_b_ap'
EVALUATION = WORK / 'eval_b_ap'
ORIGINAL_TRAIN_SHA = '1592ed3cfa972a7a535f8225ac832cb3d83ccd4e9dcd4e5db197014e5ac9c359'
LEGACY_MODULE_HASHES = {'train.py': 'dba456baf11ee118bfe86bde505fa987529990032ee81ca1fe4d591d554f00f3', 'common.py': '52272f968a549cfa4e1e1f7c99af6a94c9e4cc3ae7428e680234405db4afbcd5', 'config.py': '191837714ff38a0fb0aee43c8b326a8183d227cf59bd9642306dc237ed02194f', 'data.py': '0a8ca22ff02b882098b36bcbc727cf6f74017c2ee4c28627f212aacacadd5ea8', 'model.py': '7f1934520b8c648ab9b081b377cc594213a6f790c937326dffff7ad24a84fec9'}
