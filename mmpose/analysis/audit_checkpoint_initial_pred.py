import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


CONFIG = (
    'projects/cigpose/body_2d_keypoint/'
    'cigpose-m_1xb384-420e_coco-256x192.py'
)

CHECKPOINT = '../checkpoints/cigpose-m_coco_256x192.pth'


def find_key_by_suffix(state_dict, suffix):
    matches = [k for k in state_dict.keys() if k.endswith(suffix)]

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f'WARNING: multiple matches for {suffix}:')
        for k in matches:
            print('  ', k)

    return matches[0]


def tensor_info(name, tensor):
    print(f'\n{name}')
    print('  shape :', tuple(tensor.shape))
    print('  mean  :', tensor.float().mean().item())
    print('  std   :', tensor.float().std().item())
    print('  min   :', tensor.float().min().item())
    print('  max   :', tensor.float().max().item())


def main():
    print('Loading raw checkpoint...')

    checkpoint = torch.load(
        CHECKPOINT,
        map_location='cpu'
    )

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print('Checkpoint contains state_dict.')
    else:
        state_dict = checkpoint
        print('Checkpoint itself is treated as state_dict.')

    print()
    print('========== 1. CHECKPOINT KEYS ==========')

    target_suffixes = [
        'head.initial_pred_x.weight',
        'head.initial_pred_x.bias',
        'head.initial_pred_y.weight',
        'head.initial_pred_y.bias',
    ]

    ckpt_keys = {}

    for suffix in target_suffixes:
        key = find_key_by_suffix(state_dict, suffix)
        ckpt_keys[suffix] = key

        if key is None:
            print(f'MISSING: {suffix}')
        else:
            print(f'FOUND: {key}')
            tensor_info(
                'checkpoint tensor',
                state_dict[key]
            )

    print()
    print('========== 2. BUILD MODEL ==========')

    cfg = Config.fromfile(CONFIG)
    cfg.work_dir = 'work_dirs/challenge1_checkpoint_audit'

    runner = Runner.from_cfg(cfg)
    model = runner.model

    # Keep randomly initialized values before checkpoint loading
    before = {
        'head.initial_pred_x.weight':
            model.head.initial_pred_x.weight.detach().cpu().clone(),

        'head.initial_pred_x.bias':
            model.head.initial_pred_x.bias.detach().cpu().clone(),

        'head.initial_pred_y.weight':
            model.head.initial_pred_y.weight.detach().cpu().clone(),

        'head.initial_pred_y.bias':
            model.head.initial_pred_y.bias.detach().cpu().clone(),
    }

    print()
    print('========== 3. LOAD CHECKPOINT ==========')

    load_checkpoint(
        model,
        CHECKPOINT,
        map_location='cpu',
        strict=False
    )

    after = {
        'head.initial_pred_x.weight':
            model.head.initial_pred_x.weight.detach().cpu(),

        'head.initial_pred_x.bias':
            model.head.initial_pred_x.bias.detach().cpu(),

        'head.initial_pred_y.weight':
            model.head.initial_pred_y.weight.detach().cpu(),

        'head.initial_pred_y.bias':
            model.head.initial_pred_y.bias.detach().cpu(),
    }

    print()
    print('========== 4. BEFORE vs AFTER LOAD ==========')

    for suffix in target_suffixes:

        diff = (
            after[suffix] - before[suffix]
        ).abs()

        print()
        print(suffix)
        print(
            '  changed after loading:',
            bool(diff.max().item() > 0)
        )
        print(
            '  max abs change:',
            diff.max().item()
        )

    print()
    print('========== 5. MODEL vs CHECKPOINT ==========')

    for suffix in target_suffixes:

        ckpt_key = ckpt_keys[suffix]

        print()
        print(suffix)

        if ckpt_key is None:
            print('  Cannot compare: key missing in checkpoint.')
            continue

        ckpt_tensor = state_dict[ckpt_key].detach().cpu()
        model_tensor = after[suffix]

        diff = (
            model_tensor - ckpt_tensor
        ).abs()

        print(
            '  exact/allclose match:',
            torch.allclose(
                model_tensor,
                ckpt_tensor,
                atol=0,
                rtol=0
            )
        )

        print(
            '  max abs difference:',
            diff.max().item()
        )

    print()
    print('===========================================')


if __name__ == '__main__':
    main()
