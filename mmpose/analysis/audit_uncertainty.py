import csv
from collections import Counter

import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


CONFIG = (
    'projects/cigpose/body_2d_keypoint/'
    'cigpose-m_1xb384-420e_coco-256x192.py'
)

CHECKPOINT = '../checkpoints/cigpose-m_coco_256x192.pth'

MAX_PERSONS = 100

JOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]


def main():
    cfg = Config.fromfile(CONFIG)

    cfg.work_dir = 'work_dirs/challenge1_uncertainty_audit'

    cfg.test_dataloader.batch_size = 8
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    runner = Runner.from_cfg(cfg)

    model = runner.model

    load_checkpoint(
        model,
        CHECKPOINT,
        map_location='cpu'
    )

    model.cuda()
    model.eval()

    # SimCC output dimensions
    W = int(
        model.head.input_size[0]
        * model.head.simcc_split_ratio
    )

    H = int(
        model.head.input_size[1]
        * model.head.simcc_split_ratio
    )

    # If softmax is completely uniform:
    # max softmax(x) = 1/W
    # max softmax(y) = 1/H
    uniform_u = 1.0 - (
        ((1.0 / W) + (1.0 / H)) / 2.0
    )

    rows = []
    count = 0

    with torch.no_grad():

        for data_batch in runner.test_dataloader:

            original_samples = data_batch['data_samples']

            data = model.data_preprocessor(
                data_batch,
                training=False
            )

            inputs = data['inputs']

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cuda()

            feats = model.extract_feat(inputs)

            # Reproduce exactly the feature path used
            # before CIGPose intervention selection.
            x = model.head.final_layer(feats[-1])
            x = torch.flatten(x, 2)

            f_initial = model.head.mlp_pre_gau(x)
            f_kpts = model.head.gau(f_initial)

            h_x = model.head.initial_pred_x(f_kpts)
            h_y = model.head.initial_pred_y(f_kpts)

            conf_x = (
                h_x.softmax(dim=-1)
                .max(dim=-1)[0]
            )

            conf_y = (
                h_y.softmax(dim=-1)
                .max(dim=-1)[0]
            )

            confidence = (conf_x + conf_y) / 2.0

            uncertainty = 1.0 - confidence

            uncertainty = (
                uncertainty
                .detach()
                .cpu()
                .numpy()
            )

            conf_x = conf_x.detach().cpu().numpy()
            conf_y = conf_y.detach().cpu().numpy()

            for b in range(uncertainty.shape[0]):

                u = uncertainty[b]

                sorted_u = np.sort(u)[::-1]

                top1 = int(np.argmax(u))

                rows.append({
                    'img_id':
                        original_samples[b].metainfo[
                            'img_id'
                        ],

                    'max_u':
                        float(u.max()),

                    'min_u':
                        float(u.min()),

                    'mean_u':
                        float(u.mean()),

                    'std_u':
                        float(u.std()),

                    'range_u':
                        float(u.max() - u.min()),

                    'top1_top2':
                        float(
                            sorted_u[0]
                            - sorted_u[1]
                        ),

                    'top1_joint':
                        top1,

                    'top1_joint_name':
                        JOINT_NAMES[top1],

                    'mean_conf_x':
                        float(conf_x[b].mean()),

                    'mean_conf_y':
                        float(conf_y[b].mean()),
                })

                count += 1

                if count >= MAX_PERSONS:
                    break

            if count >= MAX_PERSONS:
                break

    out = (
        'analysis/results/'
        'challenge1_uncertainty_audit.csv'
    )

    with open(out, 'w', newline='') as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    means = np.array(
        [r['mean_u'] for r in rows]
    )

    stds = np.array(
        [r['std_u'] for r in rows]
    )

    ranges = np.array(
        [r['range_u'] for r in rows]
    )

    margins = np.array(
        [r['top1_top2'] for r in rows]
    )

    top1_counts = Counter(
        r['top1_joint_name']
        for r in rows
    )

    print()
    print(
        '========== Challenge 1 '
        'Uncertainty Audit =========='
    )

    print(f'Persons: {len(rows)}')

    print()
    print(f'SimCC W = {W}')
    print(f'SimCC H = {H}')

    print(
        'Uniform-softmax theoretical U = '
        f'{uniform_u:.9f}'
    )

    print()
    print(
        f'Mean of mean(U)  = '
        f'{means.mean():.9f}'
    )

    print(
        f'Median std(U)    = '
        f'{np.median(stds):.9f}'
    )

    print(
        f'Mean std(U)      = '
        f'{stds.mean():.9f}'
    )

    print(
        f'Median range(U)  = '
        f'{np.median(ranges):.9f}'
    )

    print(
        f'Mean range(U)    = '
        f'{ranges.mean():.9f}'
    )

    print(
        f'Median Top1-Top2 = '
        f'{np.median(margins):.9f}'
    )

    print(
        f'Mean Top1-Top2   = '
        f'{margins.mean():.9f}'
    )

    print()
    print('Top-1 joint frequency:')

    for joint, n in top1_counts.most_common():
        print(
            f'  {joint:<16} '
            f'{n:>3} / {len(rows)}'
        )

    print()
    print('Saved:', out)

    print(
        '=========================================='
    )


if __name__ == '__main__':
    main()
