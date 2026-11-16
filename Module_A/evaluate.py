"""单独验证训练好的 MLP；所有检测都进入原来的统一评估入口。"""
import argparse
import json
from pathlib import Path
import torch

import common
import config
from model import MLP
from a_contract import CHECKPOINT_SCHEMA, INPUT_SCHEMA
from a_runtime import EXPECTED_CHECKPOINT, load_frozen_base, require_val_cache, setup
from detection_cache import ReferenceCache
from predict_a import validate_a
from shared_cache import hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=config.OUTPUT / 'best.pt')
    parser.add_argument('--output', type=Path, default=config.OUTPUT / 'standalone-val')
    parser.add_argument('--device', default=config.DEVICE)
    args = parser.parse_args()
    setup(config.SEED)

    # 1. 读取本项目训练得到的 checkpoint，并检查模型和数据协议。
    saved = torch.load(args.checkpoint, map_location='cpu')
    if (saved.get('schema_version') != CHECKPOINT_SCHEMA
            or saved.get('input_schema') != INPUT_SCHEMA
            or saved.get('trained') is not True or saved.get('epoch', 0) < 1
            or saved.get('base_checkpoint_sha256') != EXPECTED_CHECKPOINT):
        raise ValueError('需要本项目正式训练得到的 A checkpoint')
    model = MLP(**saved['model_config']).to(args.device)
    model.load_state_dict(saved['model_state'])
    if any(not torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError('模型包含非法数值')
    if (model.scalar_std <= 0).any():
        raise ValueError('标准化参数无效')

    # 2. 完整 val 检测缓存，不能根据 GT 预先删除误检。
    cache = ReferenceCache(config.VAL_CACHE)
    require_val_cache(cache)
    cache.verify(True)
    val_ids = {int(x['id']) for x in json.loads(config.VAL_ANN.read_text())['images']}
    if set(saved['train_image_ids']).intersection(val_ids):
        raise ValueError('train 和 val 图片有重叠')

    # 3. A 选关节，固定 alpha=1、始终接受，然后计算完整 val AP/AR。
    base = load_frozen_base(config.BASE_MODEL, args.device)
    result = validate_a(model.eval(), base, cache, config.VAL_ANN, args.output,
                        args.device, dict(a_checkpoint_sha256=hash_file(args.checkpoint),
                                          epoch=saved['epoch']))
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
