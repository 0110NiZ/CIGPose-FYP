"""Legacy TF32 precision with the user-approved deterministic stabilization.

No import-time mutation. Call configure_legacy_precision at every GPU entry.
Deterministic cuDNN is distinct from global deterministic algorithms.
"""
import copy
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

_PROFILE = dict(
    profile_id='cigpose-legacy-cache-fp32-tf32conv-deterministic-v2',
    model_dtype='float32', autocast=False,
    matmul_allow_tf32=False, cudnn_allow_tf32=True,
    cudnn_benchmark=False, cudnn_deterministic=True,
    deterministic_algorithms=True, cublas_workspace_config=':4096:8',
    base_model_frozen=True, base_model_eval=True, flip_test=False,
    input_size_wh=[192, 256], simcc_split_ratio=2.0, use_dark=False,
    inference_batch_size=32,
    inference_batch_rule='original source shard order and batch; do not merge 17 actions',
    p0_reference='source_cache.P_0_coords',
    candidate_decoder='public head.decode and shared_cache.restore_coords',
    supervision_oks_dtype='float64',
)


def legacy_profile():
    return copy.deepcopy(_PROFILE)


def profile_fingerprint():
    return hashlib.sha256(json.dumps(_PROFILE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_profile(profile, fingerprint):
    if profile != _PROFILE or fingerprint != profile_fingerprint():
        raise ValueError('Precision profile differs from approved legacy-cache protocol')


def assert_legacy_precision():
    actual = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    wrong = [k for k,v in actual.items() if v != _PROFILE[k]]
    if wrong:
        raise RuntimeError('Call configure_legacy_precision; mismatched flags: ' + ', '.join(wrong))
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != _PROFILE['cublas_workspace_config']:
        raise RuntimeError('Deterministic legacy precision requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise RuntimeError('Legacy-cache inference must run outside autocast')
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError('Legacy-cache inference requires default float32')


def runtime_record():
    assert_legacy_precision()
    return dict(precision_profile=legacy_profile(), precision_profile_sha256=profile_fingerprint(),
                torch_version=torch.__version__, numpy_version=np.__version__, python=sys.version,
                cuda_build=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                threads=torch.get_num_threads(), cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def configure_legacy_precision(seed=42, threads=4):
    if seed < 0 or threads < 1:
        raise ValueError('Invalid seed or thread count')
    desired = _PROFILE['cublas_workspace_config']
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != desired:
        raise RuntimeError('Initialize the shared precision policy before CUDA; restart this process')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = desired
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    assert_legacy_precision()
    return runtime_record()
