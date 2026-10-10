import argparse
import csv
import json
import math
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner


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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        default='projects/cigpose/body_2d_keypoint/'
                'cigpose-m_1xb384-420e_coco-256x192.py'
    )

    parser.add_argument(
        '--checkpoint',
        default='../checkpoints/cigpose-m_coco_256x192.pth'
    )

    parser.add_argument(
        '--max-persons',
        type=int,
        default=None
    )

    # 17 次 intervention 比较重，先保守一点
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--output-dir',
        default='analysis/results'
    )

    return parser.parse_args()


def unwrap_dataset_cfg(dataset_cfg):
    current = dataset_cfg

    while 'ann_file' not in current and 'dataset' in current:
        current = current['dataset']

    return current


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = float(values.mean())

    if len(values) == 1:
        return mean, np.nan, np.nan

    se = float(values.std(ddof=1) / math.sqrt(len(values)))

    return (
        mean,
        mean - 1.96 * se,
        mean + 1.96 * se
    )


def decode_simcc_to_original(
    pred_x,
    pred_y,
    input_center,
    input_scale,
    input_size,
):
    """
    pred_x: [K, Wx]
    pred_y: [K, Hy]

    return:
        [K, 2] original-image coordinates
    """

    input_w = float(input_size[0])
    input_h = float(input_size[1])

    bins_x = pred_x.shape[-1]
    bins_y = pred_y.shape[-1]

    split_x = bins_x / input_w
    split_y = bins_y / input_h

    x_idx = pred_x.argmax(dim=-1).float()
    y_idx = pred_y.argmax(dim=-1).float()

    x_input = x_idx / split_x
    y_input = y_idx / split_y

    center_x = float(input_center[0])
    center_y = float(input_center[1])

    scale_x = float(input_scale[0])
    scale_y = float(input_scale[1])

    x_orig = (
        (x_input / input_w - 0.5) * scale_x
        + center_x
    )

    y_orig = (
        (y_input / input_h - 0.5) * scale_y
        + center_y
    )

    coords = torch.stack(
        [x_orig, y_orig],
        dim=-1
    )

    return coords.detach().cpu().numpy()


def unpack_xy(output):
    """
    diagnostic_forward 里的 baseline / forced prediction
    预期都是 (pred_x, pred_y)。

    写得稍微稳一点，方便报错。
    """
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output[0], output[1]

    if isinstance(output, dict):
        if 'pred_x' in output and 'pred_y' in output:
            return output['pred_x'], output['pred_y']

    raise RuntimeError(
        f'Unexpected diagnostic output format: {type(output)}'
    )


def pose_error(
    pred_xy,
    gt_xy,
    labeled_mask,
    bbox,
):
    """
    整个人的 pose error。

    只在有 GT 的 joint (v=1/v=2) 上算，
    再用 sqrt(bbox_w * bbox_h) 做 person scale normalization。
    """

    bbox_w = float(bbox[2])
    bbox_h = float(bbox[3])

    norm = math.sqrt(
        max(bbox_w * bbox_h, 1e-12)
    )

    joint_errors = np.linalg.norm(
        pred_xy - gt_xy,
        axis=1
    ) / norm

    return float(
        joint_errors[labeled_mask].mean()
    )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 76)
    print('Challenge 1 - Utility Gap Audit')
    print('Random Top-1 vs Official Top-1 vs Oracle Top-1')
    print('=' * 76)

    # ---------------------------------------------------------
    # 1. Config
    # ---------------------------------------------------------
    cfg = Config.fromfile(args.config)

    cfg.load_from = None

    cfg.work_dir = (
        'work_dirs/challenge1_utility_gap'
    )

    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    # 用 GT person bbox
    dataset_cfg = unwrap_dataset_cfg(
        cfg.test_dataloader.dataset
    )

    if 'bbox_file' in dataset_cfg:
        dataset_cfg['bbox_file'] = None

    data_root = dataset_cfg.get('data_root', '')
    ann_file = dataset_cfg['ann_file']

    if os.path.isabs(ann_file):
        ann_path = ann_file
    else:
        ann_path = os.path.join(
            data_root,
            ann_file
        )

    ann_path = os.path.normpath(ann_path)

    print('Annotation:', ann_path)

    # ---------------------------------------------------------
    # 2. Raw COCO annotations
    # ---------------------------------------------------------
    with open(
        ann_path,
        'r',
        encoding='utf-8'
    ) as f:
        coco = json.load(f)

    anns_by_id = {
        int(ann['id']): ann
        for ann in coco['annotations']
    }

    # ---------------------------------------------------------
    # 3. Runner / model
    # ---------------------------------------------------------
    runner = Runner.from_cfg(cfg)

    runner.call_hook('before_run')

    print('Loading checkpoint...')
    runner.load_checkpoint(args.checkpoint)

    model = runner.model

    if hasattr(model, 'module'):
        model = model.module

    model.eval()

    dataloader = runner.test_dataloader
    dataset = dataloader.dataset

    print('Dataset persons:', len(dataset))

    # ---------------------------------------------------------
    # 4. Results
    # ---------------------------------------------------------
    rows = []

    random_gains = []
    official_gains = []
    oracle_gains = []
    oracle_abstain_gains = []

    official_oracle_gaps = []

    official_harm = []
    random_harm_prob = []

    official_oracle_match = []

    baseline_errors = []

    global_index = 0

    # ---------------------------------------------------------
    # 5. Loop
    # ---------------------------------------------------------
    with torch.no_grad():

        for batch_idx, data in enumerate(dataloader):

            if (
                args.max_persons is not None
                and global_index >= args.max_persons
            ):
                break

            processed = model.data_preprocessor(
                data,
                training=False
            )

            feats = model.extract_feat(
                processed['inputs']
            )

            # 一次得到：
            # - official uncertainty Top-1
            # - no-intervention baseline
            # - 17 个 forced intervention outputs
            diag = model.head.diagnostic_forward(feats)

            top1_indices = (
                diag['top1_indices']
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            baseline_x, baseline_y = unpack_xy(
                diag['baseline']
            )

            forced_outputs = diag['forced']

            if len(forced_outputs) != 17:
                raise RuntimeError(
                    f'Expected 17 forced outputs, '
                    f'got {len(forced_outputs)}'
                )

            forced_xy = [
                unpack_xy(out)
                for out in forced_outputs
            ]

            data_samples = processed['data_samples']
            batch_n = len(data_samples)

            for b in range(batch_n):

                if (
                    args.max_persons is not None
                    and global_index >= args.max_persons
                ):
                    break

                sample = data_samples[b]

                info = dataset.get_data_info(
                    global_index
                )

                ann_id = info.get('id', None)

                if ann_id is None:
                    raise RuntimeError(
                        f'No annotation id at index '
                        f'{global_index}'
                    )

                raw_ann = anns_by_id.get(
                    int(ann_id),
                    None
                )

                if raw_ann is None:
                    raise RuntimeError(
                        f'Cannot find COCO ann_id={ann_id}'
                    )

                raw_kpts = np.asarray(
                    raw_ann['keypoints'],
                    dtype=np.float32
                ).reshape(17, 3)

                gt_xy = raw_kpts[:, :2]
                visibility = raw_kpts[:, 2].astype(
                    np.int64
                )

                # 有 GT 的关节
                labeled_mask = visibility > 0

                if labeled_mask.sum() == 0:
                    global_index += 1
                    continue

                meta = sample.metainfo

                input_center = np.asarray(
                    meta['input_center'],
                    dtype=np.float32
                )

                input_scale = np.asarray(
                    meta['input_scale'],
                    dtype=np.float32
                )

                input_size = np.asarray(
                    meta['input_size'],
                    dtype=np.float32
                )

                # -------------------------------------------------
                # Baseline: 完全不 intervention
                # -------------------------------------------------
                baseline_xy = decode_simcc_to_original(
                    baseline_x[b],
                    baseline_y[b],
                    input_center,
                    input_scale,
                    input_size,
                )

                base_error = pose_error(
                    baseline_xy,
                    gt_xy,
                    labeled_mask,
                    raw_ann['bbox'],
                )

                baseline_errors.append(base_error)

                # -------------------------------------------------
                # 17 个 forced intervention
                # -------------------------------------------------
                gains = np.zeros(
                    17,
                    dtype=np.float64
                )

                corrected_errors = np.zeros(
                    17,
                    dtype=np.float64
                )

                for joint_id in range(17):

                    pred_x_j, pred_y_j = (
                        forced_xy[joint_id]
                    )

                    corrected_xy = (
                        decode_simcc_to_original(
                            pred_x_j[b],
                            pred_y_j[b],
                            input_center,
                            input_scale,
                            input_size,
                        )
                    )

                    corrected_error = pose_error(
                        corrected_xy,
                        gt_xy,
                        labeled_mask,
                        raw_ann['bbox'],
                    )

                    corrected_errors[joint_id] = (
                        corrected_error
                    )

                    # gain > 0 = 修完更好
                    # gain < 0 = 修完更差
                    gains[joint_id] = (
                        base_error
                        - corrected_error
                    )

                # -------------------------------------------------
                # Official
                # -------------------------------------------------
                official_joint = int(
                    top1_indices[b]
                )

                official_gain = float(
                    gains[official_joint]
                )

                # -------------------------------------------------
                # Random
                #
                # 真正 selector 是在 17 个 joint 中选，
                # 所以这里 random 也必须在 17 个中均匀随机。
                #
                # 它的 expected gain = 17 个 gain 的平均。
                # -------------------------------------------------
                random_gain = float(
                    gains.mean()
                )

                # 随机选到 harmful intervention 的概率
                random_harm = float(
                    (gains < 0).mean()
                )

                # -------------------------------------------------
                # Oracle forced Top-1
                #
                # 和 Official 一样，“必须修一个”，
                # 只是 Oracle 事先知道哪个收益最大。
                # -------------------------------------------------
                oracle_joint = int(
                    np.argmax(gains)
                )

                oracle_gain = float(
                    gains[oracle_joint]
                )

                # -------------------------------------------------
                # Oracle + No Intervention
                #
                # 如果 17 个全部是负收益，
                # 最聪明的动作其实是什么都不修。
                #
                # 这个先作为额外 upper bound 报出来，
                # 不和 official 的 forced Top-1 混为一谈。
                # -------------------------------------------------
                oracle_abstain_gain = max(
                    0.0,
                    oracle_gain
                )

                gap = (
                    oracle_gain
                    - official_gain
                )

                match = int(
                    official_joint == oracle_joint
                )

                official_is_harmful = int(
                    official_gain < 0
                )

                random_gains.append(
                    random_gain
                )

                official_gains.append(
                    official_gain
                )

                oracle_gains.append(
                    oracle_gain
                )

                oracle_abstain_gains.append(
                    oracle_abstain_gain
                )

                official_oracle_gaps.append(
                    gap
                )

                official_harm.append(
                    official_is_harmful
                )

                random_harm_prob.append(
                    random_harm
                )

                official_oracle_match.append(
                    match
                )

                row = {
                    'dataset_index':
                        global_index,

                    'ann_id':
                        int(raw_ann['id']),

                    'img_id':
                        int(raw_ann['image_id']),

                    'baseline_pose_error':
                        base_error,

                    'official_joint_id':
                        official_joint,

                    'official_joint_name':
                        JOINT_NAMES[official_joint],

                    'official_gain':
                        official_gain,

                    'random_expected_gain':
                        random_gain,

                    'oracle_joint_id':
                        oracle_joint,

                    'oracle_joint_name':
                        JOINT_NAMES[oracle_joint],

                    'oracle_gain':
                        oracle_gain,

                    'oracle_abstain_gain':
                        oracle_abstain_gain,

                    'oracle_minus_official':
                        gap,

                    'official_equals_oracle':
                        match,

                    'official_harmful':
                        official_is_harmful,

                    'random_harm_probability':
                        random_harm,
                }

                # 后面训练 Utility Selector 时，
                # 这 17 个 gain 就是监督标签。
                for j in range(17):
                    row[
                        f'gain_{j}_{JOINT_NAMES[j]}'
                    ] = float(gains[j])

                rows.append(row)

                global_index += 1

            if batch_idx % 20 == 0:
                print(
                    f'Processed persons: '
                    f'{global_index}'
                )

    # ---------------------------------------------------------
    # 6. Statistics
    # ---------------------------------------------------------
    random_gains = np.asarray(
        random_gains,
        dtype=np.float64
    )

    official_gains = np.asarray(
        official_gains,
        dtype=np.float64
    )

    oracle_gains = np.asarray(
        oracle_gains,
        dtype=np.float64
    )

    oracle_abstain_gains = np.asarray(
        oracle_abstain_gains,
        dtype=np.float64
    )

    gaps = np.asarray(
        official_oracle_gaps,
        dtype=np.float64
    )

    # Official - Random
    official_vs_random = (
        official_gains
        - random_gains
    )

    ovr_mean, ovr_low, ovr_high = mean_ci95(
        official_vs_random
    )

    # Oracle - Official
    gap_mean, gap_low, gap_high = mean_ci95(
        gaps
    )

    mean_random = float(
        random_gains.mean()
    )

    mean_official = float(
        official_gains.mean()
    )

    mean_oracle = float(
        oracle_gains.mean()
    )

    mean_oracle_abstain = float(
        oracle_abstain_gains.mean()
    )

    official_harm_rate = float(
        np.mean(official_harm)
    )

    random_harm_rate = float(
        np.mean(random_harm_prob)
    )

    oracle_forced_harm_rate = float(
        np.mean(oracle_gains < 0)
    )

    match_rate = float(
        np.mean(official_oracle_match)
    )

    # Official 在 17 个 intervention 中的平均 rank
    # rank 1 = 最好
    official_ranks = []

    for row in rows:

        gains = np.asarray([
            row[f'gain_{j}_{JOINT_NAMES[j]}']
            for j in range(17)
        ])

        official_joint = row['official_joint_id']

        order = np.argsort(-gains)

        rank = (
            np.where(
                order == official_joint
            )[0][0]
            + 1
        )

        official_ranks.append(rank)

    mean_official_rank = float(
        np.mean(official_ranks)
    )

    median_official_rank = float(
        np.median(official_ranks)
    )

    # ---------------------------------------------------------
    # 7. Print
    # ---------------------------------------------------------
    print()
    print('=' * 76)
    print('RESULT')
    print('=' * 76)

    print(
        f'Persons evaluated               = '
        f'{len(rows)}'
    )

    print()
    print('Mean pose-level refinement gain:')
    print(
        f'Random Top-1 expected gain       = '
        f'{mean_random:+.8f}'
    )

    print(
        f'Official uncertainty Top-1       = '
        f'{mean_official:+.8f}'
    )

    print(
        f'Oracle forced Top-1              = '
        f'{mean_oracle:+.8f}'
    )

    print(
        f'Oracle + No Intervention         = '
        f'{mean_oracle_abstain:+.8f}'
    )

    print()
    print('Official - Random:')
    print(
        f'Mean difference                  = '
        f'{ovr_mean:+.8f}'
    )

    print(
        f'95% CI                           = '
        f'[{ovr_low:+.8f}, '
        f'{ovr_high:+.8f}]'
    )

    print()
    print('Oracle - Official:')
    print(
        f'Mean utility gap                 = '
        f'{gap_mean:+.8f}'
    )

    print(
        f'95% CI                           = '
        f'[{gap_low:+.8f}, '
        f'{gap_high:+.8f}]'
    )

    print()
    print('Selection quality:')
    print(
        f'Official exactly matches Oracle  = '
        f'{match_rate * 100:.2f}%'
    )

    print(
        f'Official mean utility rank / 17  = '
        f'{mean_official_rank:.2f}'
    )

    print(
        f'Official median utility rank     = '
        f'{median_official_rank:.1f}'
    )

    print()
    print('Harm rate (gain < 0):')
    print(
        f'Random expected harm rate        = '
        f'{random_harm_rate * 100:.2f}%'
    )

    print(
        f'Official Top-1 harm rate         = '
        f'{official_harm_rate * 100:.2f}%'
    )

    print(
        f'Oracle forced Top-1 harm rate    = '
        f'{oracle_forced_harm_rate * 100:.2f}%'
    )

    print()
    print('=' * 76)
    print('INTERPRETATION')
    print('=' * 76)

    if ovr_low > 0:
        print(
            '1. Official selector is better than '
            'random in refinement utility.'
        )
    elif ovr_high < 0:
        print(
            '1. Official selector is worse than '
            'random in refinement utility.'
        )
    else:
        print(
            '1. Official selector does not clearly '
            'beat random in refinement utility.'
        )

    if gap_low > 0:
        print(
            '2. Oracle is significantly better than '
            'Official: there is selection headroom.'
        )
    else:
        print(
            '2. Current data does not yet show a '
            'clear Oracle-vs-Official utility gap.'
        )

    # ---------------------------------------------------------
    # 8. Save per-person 17 utility labels
    #
    # 这个不是“实验日志”。
    # 后面训练 MLP / Transformer Utility Selector
    # 正好可以直接拿这里的 gain 当监督标签。
    # ---------------------------------------------------------
    csv_path = os.path.join(
        args.output_dir,
        'challenge1_coco_utility_gap.csv'
    )

    if len(rows) > 0:
        with open(
            csv_path,
            'w',
            newline='',
            encoding='utf-8'
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

    print()
    print(
        'Saved utility labels to: '
        f'{csv_path}'
    )


if __name__ == '__main__':
    main()
