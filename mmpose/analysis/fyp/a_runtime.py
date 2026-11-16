"""Runtime/provenance helpers. The public CIGPose head and evaluator stay unchanged."""
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from a_contract import CHECKPOINT_SCHEMA, INPUT_SCHEMA
from build_shared_cache import EXPECTED_CHECKPOINT
from gain_predictor import GainPredictor
from shared_cache import hash_file
from precision_policy import configure_legacy_precision, assert_legacy_precision, require_profile

ROOT = Path(__file__).resolve().parents[3]
MMPOSE = ROOT / 'mmpose'
HEAD_REL = 'mmpose/mmpose/models/heads/coord_cls_heads/cig_head.py'
VAL_DETECTION_SHA = '53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59'


def setup(seed=42, threads=4):
    return configure_legacy_precision(seed, threads)


def require_cache_model(cache):
    p = cache.run['protocol']
    if 'precision_profile' in p:
        require_profile(p['precision_profile'], p.get('precision_profile_sha256'))
    if (p.get('checkpoint_sha256') != EXPECTED_CHECKPOINT
            or p.get('flip_test') is not False or p.get('base_model_eval') is not True
            or p.get('precision') != 'float32'):
        raise ValueError('Reference cache does not use the fixed frozen/single-view model')
    hashes = p.get('code_hashes', p.get('implementation_sha256', {}))
    if hashes.get(HEAD_REL) != hash_file(ROOT / HEAD_REL):
        raise ValueError('Public head implementation differs from cache; review before proceeding')


def require_val_cache(cache):
    require_cache_model(cache)
    p = cache.run['protocol']
    if (cache.run.get('schema_version') != 'cigpose-detection-cache-v1'
            or p.get('dataset_split') != 'val2017'
            or p.get('source_detection_sha256') != VAL_DETECTION_SHA
            or not p.get('source_annotation_sha256')
            or p.get('sample_only') is not False
            or len(cache.expected) != 104125):
        raise ValueError('Formal validation requires the approved full 104125-detection val cache')


def load_frozen_base(checkpoint, device):
    assert_legacy_precision()
    checkpoint = Path(checkpoint).resolve()
    if hash_file(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('Wrong base CIGPose checkpoint')
    sys.path.insert(0, str(MMPOSE))
    from mmengine.config import Config
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    register_all_modules()
    cfg_path = MMPOSE / 'projects/cigpose/body_2d_keypoint/cigpose-m_coco-256x192_detection-single-view.py'
    cwd = Path.cwd()
    try:
        os.chdir(MMPOSE)
        cfg = Config.fromfile(str(cfg_path))
        if cfg.model.test_cfg.flip_test is not False or tuple(cfg.codec.input_size) != (192, 256):
            raise ValueError('Unexpected base model configuration')
        model = init_model(cfg, str(checkpoint), device=device)
    finally:
        os.chdir(cwd)
    return model.requires_grad_(False).eval()


def load_a_checkpoint(path, device):
    # Checkpoints are local trusted project artifacts, not externally supplied models.
    payload = torch.load(path, map_location='cpu')
    if (payload.get('schema_version') != CHECKPOINT_SCHEMA
            or payload.get('input_schema') != INPUT_SCHEMA
            or payload.get('trained') is not True or payload.get('epoch', 0) < 1
            or payload.get('base_checkpoint_sha256') != EXPECTED_CHECKPOINT):
        raise ValueError('Expected a trained Module A checkpoint with matching provenance')
    model = GainPredictor(**payload['model_config']).to(device)
    model.load_state_dict(payload['model_state'])
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite Module A parameters')
    if (model.scalar_std <= 0).any():
        raise ValueError('Invalid saved normalization statistics')
    return model.eval(), payload


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)
