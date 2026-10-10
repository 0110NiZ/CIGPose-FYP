import argparse
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

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32
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

    mean = values.mean()

    if len(values) == 1:
        return mean, np.nan, np.nan

    se = values.std(ddof=1) / math.sqrt(len(values))

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

    返回原图坐标 [K, 2]
    """

    input_w = float(input_size[0])
    input_h = float(input_size[1])

    bins_x = pred_x.shape[-1]
    bins_y = pred_y.shape[-1]

    split_x = bins_x / input_w
    split_y = bins_y / input_h

    x_idx = pred_x.argmax(dim=-1).float()
    y_idx = pred_y.argmax(dim=-1).float()

    # SimCC coordinate -> network input coordinate
    x_input = x_idx / split_x
    y_input = y_idx / split_y

    center_x = float(input_center[0])
    center_y = float(input_center[1])

    scale_x = float(input_scale[0])
    scale_y = float(input_scale[1])

    # network input coordinate -> original image coordinate
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

    return coords.cpu().numpy()


def main():
    args = parse_args()

    print('=' * 72)
    print('Challenge 1 - COCO Uncertainty vs Localization Error')
    print('=' * 72)

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------
    cfg = Config.fromfile(args.config)

    cfg.load_from = None

    cfg.work_dir = (
        'work_dirs/challenge1_uncertainty_vs_error'
    )

    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    # 使用 GT person bbox
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
    # Raw COCO annotation
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
    # Runner / model
    # ---------------------------------------------------------
    runner = Runner.from_cfg(cfg)

    # 初始化 EMAHook，避免 checkpoint load 报错
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
    # Results
    # ---------------------------------------------------------
    top1_errors = []
    random_expected_errors = []
    paired_diffs = []

    top1_is_worse_than_random_mean = []

    total_persons = 0
    evaluable_persons = 0
    top1_unlabeled = 0

    top1_visibility_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    global_index = 0

    # ---------------------------------------------------------
    # Loop
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

            # 我们之前已经验证过这个 diagnostic_forward
            diag = model.head.diagnostic_forward(feats)

            top1_indices = (
                diag['top1_indices']
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            # baseline = 不做 intervention 的最终预测
            baseline = diag['baseline']

            if (
                isinstance(baseline, (tuple, list))
                and len(baseline) == 2
            ):
                baseline_x = baseline[0]
                baseline_y = baseline[1]

            else:
                raise RuntimeError(
                    'diag["baseline"] 的格式和预期不同。'
                    f' type={type(baseline)}'
                )

            data_samples = processed['data_samples']

            batch_n = len(data_samples)

            for b in range(batch_n):

                if (
                    args.max_persons is not None
                    and global_index >= args.max_persons
                ):
                    break

                total_persons += 1

                sample = data_samples[b]

                # -------------------------------------------------
                # 找对应 raw COCO annotation
                # -------------------------------------------------
                info = dataset.get_data_info(
                    global_index
                )

                ann_id = info.get('id', None)

                if ann_id is None:
                    raise RuntimeError(
                        f'index {global_index} '
                        '没有 annotation id'
                    )

                ann_id = int(ann_id)

                raw_ann = anns_by_id.get(
                    ann_id,
                    None
                )

                if raw_ann is None:
                    raise RuntimeError(
                        f'找不到 annotation id={ann_id}'
                    )

                raw_kpts = np.asarray(
                    raw_ann['keypoints'],
                    dtype=np.float32
                ).reshape(17, 3)

                gt_xy = raw_kpts[:, :2]
                visibility = (
                    raw_kpts[:, 2]
                    .astype(np.int64)
                )

                labeled_mask = visibility > 0

                if labeled_mask.sum() == 0:
                    global_index += 1
                    continue

                # -------------------------------------------------
                # decode baseline prediction
                # -------------------------------------------------
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

                pred_xy = decode_simcc_to_original(
                    baseline_x[b],
                    baseline_y[b],
                    input_center,
                    input_scale,
                    input_size,
                )

                # -------------------------------------------------
                # normalized localization error
                #
                # 用 sqrt(bbox_w * bbox_h) 做尺度归一化，
                # 防止大人/小人不能直接比较。
                # -------------------------------------------------
                bbox = raw_ann['bbox']

                bbox_w = float(bbox[2])
                bbox_h = float(bbox[3])

                norm = math.sqrt(
                    max(bbox_w * bbox_h, 1e-12)
                )

                errors = np.linalg.norm(
                    pred_xy - gt_xy,
                    axis=1
                ) / norm

                # -------------------------------------------------
                # Official uncertainty Top-1
                # -------------------------------------------------
                top1_joint = int(
                    top1_indices[b]
                )

                top1_v = int(
                    visibility[top1_joint]
                )

                top1_visibility_counts[top1_v] += 1

                # v=0 没有可用 GT，
                # 无法判断真实定位误差
                if top1_v == 0:
                    top1_unlabeled += 1
                    global_index += 1
                    continue

                # -------------------------------------------------
                # Official Top-1 error
                # -------------------------------------------------
                top1_error = float(
                    errors[top1_joint]
                )

                # -------------------------------------------------
                # Random baseline
                #
                # 和前面的定义一致：
                # 只在有标签 joint (v=1/v=2) 里随机挑。
                #
                # 随机 joint 的“期望定位误差”
                # 就是这个 person 所有 labeled joints
                # 的平均 error。
                # -------------------------------------------------
                random_error = float(
                    errors[labeled_mask].mean()
                )

                diff = (
                    top1_error
                    - random_error
                )

                top1_errors.append(
                    top1_error
                )

                random_expected_errors.append(
                    random_error
                )

                paired_diffs.append(
                    diff
                )

                top1_is_worse_than_random_mean.append(
                    int(top1_error > random_error)
                )

                evaluable_persons += 1

                global_index += 1

            if batch_idx % 20 == 0:
                print(
                    f'Processed persons: '
                    f'{global_index}'
                )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    top1_errors = np.asarray(
        top1_errors,
        dtype=np.float64
    )

    random_expected_errors = np.asarray(
        random_expected_errors,
        dtype=np.float64
    )

    paired_diffs = np.asarray(
        paired_diffs,
        dtype=np.float64
    )

    diff_mean, diff_low, diff_high = (
        mean_ci95(paired_diffs)
    )

    mean_top1 = float(
        top1_errors.mean()
    )

    mean_random = float(
        random_expected_errors.mean()
    )

    median_top1 = float(
        np.median(top1_errors)
    )

    median_random = float(
        np.median(random_expected_errors)
    )

    ratio = (
        mean_top1 / mean_random
        if mean_random > 0
        else np.nan
    )

    worse_rate = float(
        np.mean(
            top1_is_worse_than_random_mean
        )
    )

    coverage = (
        evaluable_persons / total_persons
        if total_persons > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------
    print()
    print('=' * 72)
    print('RESULT')
    print('=' * 72)

    print(
        f'Total persons evaluated           = '
        f'{total_persons}'
    )

    print(
        f'Top-1 joint has GT label          = '
        f'{evaluable_persons}'
    )

    print(
        f'Top-1 labeled coverage            = '
        f'{coverage * 100:.2f}%'
    )

    print(
        f'Top-1 v=0 count                   = '
        f'{top1_unlabeled}'
    )

    print()
    print('Top-1 visibility distribution:')

    print(
        f'v=0 : {top1_visibility_counts[0]}'
    )
    print(
        f'v=1 : {top1_visibility_counts[1]}'
    )
    print(
        f'v=2 : {top1_visibility_counts[2]}'
    )

    print()
    print('Normalized localization error:')
    print(
        f'Official uncertainty Top-1 mean  = '
        f'{mean_top1:.6f}'
    )

    print(
        f'Random labeled-joint mean         = '
        f'{mean_random:.6f}'
    )

    print(
        f'Difference                        = '
        f'{diff_mean:+.6f}'
    )

    print(
        f'Official / Random                 = '
        f'{ratio:.3f}x'
    )

    print()
    print(
        f'Official Top-1 median             = '
        f'{median_top1:.6f}'
    )

    print(
        f'Random labeled-joint median       = '
        f'{median_random:.6f}'
    )

    print()
    print('Paired difference 95% CI:')
    print(
        f'[{diff_low:+.6f}, '
        f'{diff_high:+.6f}]'
    )

    print()
    print(
        'Fraction where Top-1 error > '
        'person random-mean error      = '
        f'{worse_rate * 100:.2f}%'
    )

    print()

    if diff_low > 0:
        print(
            'PASS: uncertainty Top-1 selects '
            'joints with significantly larger '
            'localization error than random.'
        )

    elif diff_high < 0:
        print(
            'RESULT: uncertainty Top-1 actually '
            'selects joints with smaller error '
            'than the random baseline.'
        )

    else:
        print(
            'NOT CLEAR: current result does not '
            'show a clear difference from random.'
        )


if __name__ == '__main__':
    main()
