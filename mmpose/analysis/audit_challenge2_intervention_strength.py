import argparse
import csv
import json
import math
import os
from collections import defaultdict

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

# α=0 是完全不修
# α=1 是原 CIGPose 的完全 canonical replacement
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


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
        default=16
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

    se = float(
        values.std(ddof=1)
        / math.sqrt(len(values))
    )

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
        (x_input / input_w - 0.5)
        * scale_x
        + center_x
    )

    y_orig = (
        (y_input / input_h - 0.5)
        * scale_y
        + center_y
    )

    coords = torch.stack(
        [x_orig, y_orig],
        dim=-1
    )

    return (
        coords
        .detach()
        .cpu()
        .numpy()
    )


def pose_error(
    pred_xy,
    gt_xy,
    labeled_mask,
    bbox,
):
    """
    整个人的 pose error。

    gain > 0:
        refinement 后整体 pose 更好

    gain < 0:
        refinement 后整体 pose 更差
    """

    bbox_w = float(bbox[2])
    bbox_h = float(bbox[3])

    norm = math.sqrt(
        max(
            bbox_w * bbox_h,
            1e-12
        )
    )

    errors = np.linalg.norm(
        pred_xy - gt_xy,
        axis=1
    ) / norm

    return float(
        errors[labeled_mask].mean()
    )


def predict_from_joint_features(
    head,
    f_kpts,
):
    """
    从 intervention 后的 joint features
    走 CIGPose 原本后半段：

        GNN
        -> final SimCC heads
    """

    g = head.gcn(f_kpts)

    pred_x = head.cls_gx(g)
    pred_y = head.cls_gy(g)

    return pred_x, pred_y


def apply_strength(
    head,
    f_kpts,
    selected_joint,
    alpha,
):
    """
    只对 CIGPose 已经选中的 Top-1 joint 动手。

    alpha = 0:
        原 feature

    alpha = 1:
        原 CIGPose 的 canonical replacement

    中间值：
        原 feature 和 canonical feature 混合
    """

    if alpha == 0:
        return f_kpts

    output = f_kpts.clone()

    batch_size = f_kpts.shape[0]

    batch_ids = torch.arange(
        batch_size,
        device=f_kpts.device
    )

    canonical = (
        head.cim.canonical_features(
            selected_joint
        )
    )

    original = (
        f_kpts[
            batch_ids,
            selected_joint
        ]
    )

    blended = (
        (1.0 - alpha) * original
        + alpha * canonical
    )

    output[
        batch_ids,
        selected_joint
    ] = blended

    return output


def main():
    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    print('=' * 82)
    print(
        'Challenge 2 - Intervention Strength Diagnostic'
    )
    print(
        'Same selected joint, different canonical replacement strength'
    )
    print('=' * 82)

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------
    cfg = Config.fromfile(
        args.config
    )

    cfg.load_from = None

    cfg.work_dir = (
        'work_dirs/'
        'challenge2_intervention_strength'
    )

    cfg.test_dataloader.batch_size = (
        args.batch_size
    )

    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    # 用 GT person bbox
    dataset_cfg = unwrap_dataset_cfg(
        cfg.test_dataloader.dataset
    )

    if 'bbox_file' in dataset_cfg:
        dataset_cfg['bbox_file'] = None

    data_root = dataset_cfg.get(
        'data_root',
        ''
    )

    ann_file = dataset_cfg['ann_file']

    if os.path.isabs(ann_file):
        ann_path = ann_file
    else:
        ann_path = os.path.join(
            data_root,
            ann_file
        )

    ann_path = os.path.normpath(
        ann_path
    )

    print(
        'Annotation:',
        ann_path
    )

    # ---------------------------------------------------------
    # Raw COCO annotations
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

    images_by_id = {
        int(img['id']): img
        for img in coco['images']
    }

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    runner = Runner.from_cfg(cfg)

    # EMAHook 先初始化
    runner.call_hook('before_run')

    print(
        'Loading checkpoint...'
    )

    runner.load_checkpoint(
        args.checkpoint
    )

    model = runner.model

    if hasattr(model, 'module'):
        model = model.module

    model.eval()

    head = model.head

    dataloader = runner.test_dataloader
    dataset = dataloader.dataset

    print(
        'Dataset persons:',
        len(dataset)
    )

    # ---------------------------------------------------------
    # 用 hook 拿到 intervention 前的 f_kpts
    #
    # 不修改 cig_head.py。
    #
    # head.forward() 调用 CIM 前，
    # CIM 收到的第一个输入就是原始 f_kpts。
    # ---------------------------------------------------------
    captured = {}

    def capture_cim_input(
        module,
        inputs
    ):
        captured['f_kpts'] = (
            inputs[0]
            .detach()
        )

    hook_handle = (
        head.cim.register_forward_pre_hook(
            capture_cim_input
        )
    )

    # ---------------------------------------------------------
    # Result storage
    # ---------------------------------------------------------
    rows = []

    gains_by_alpha = {
        alpha: []
        for alpha in ALPHAS
    }

    best_alpha_soft_counts = {
        alpha: 0.0
        for alpha in ALPHAS
    }

    alpha1_is_best = []
    oracle_minus_alpha1 = []

    selected_joint_stats = defaultdict(
        lambda: {
            'count': 0,
            'best_alpha_sum': 0.0,
            'alpha1_best': 0,
            'oracle_gap_sum': 0.0,
        }
    )

    scale_group_stats = {
        'small': {
            'count': 0,
            'best_alpha_sum': 0.0,
            'alpha1_best': 0,
            'oracle_gap_sum': 0.0,
        },
        'medium': {
            'count': 0,
            'best_alpha_sum': 0.0,
            'alpha1_best': 0,
            'oracle_gap_sum': 0.0,
        },
        'large': {
            'count': 0,
            'best_alpha_sum': 0.0,
            'alpha1_best': 0,
            'oracle_gap_sum': 0.0,
        },
    }

    # 先把所有 normalized scale 算出来，
    # 后面按三等分分组
    normalized_scale_by_ann = {}

    for ann_id, ann in anns_by_id.items():

        image = images_by_id[
            int(ann['image_id'])
        ]

        _, _, bw, bh = ann['bbox']

        iw = float(image['width'])
        ih = float(image['height'])

        scale = math.sqrt(
            max(
                (float(bw) / iw)
                * (float(bh) / ih),
                1e-12
            )
        )

        normalized_scale_by_ann[
            ann_id
        ] = scale

    # 只按 val 里实际 person 的 scale
    all_scales = []

    for i in range(len(dataset)):

        info = dataset.get_data_info(i)

        ann_id = info.get(
            'id',
            None
        )

        if ann_id is not None:

            all_scales.append(
                normalized_scale_by_ann[
                    int(ann_id)
                ]
            )

    q1 = float(
        np.quantile(
            all_scales,
            1 / 3
        )
    )

    q2 = float(
        np.quantile(
            all_scales,
            2 / 3
        )
    )

    print()
    print(
        'Normalized scale thresholds:'
    )

    print(
        f'small <= {q1:.6f}'
    )

    print(
        f'medium <= {q2:.6f}'
    )

    print(
        f'large > {q2:.6f}'
    )

    def get_scale_group(scale):

        if scale <= q1:
            return 'small'

        elif scale <= q2:
            return 'medium'

        return 'large'

    global_index = 0

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    with torch.no_grad():

        for batch_idx, data in enumerate(
            dataloader
        ):

            if (
                args.max_persons is not None
                and global_index
                >= args.max_persons
            ):
                break

            processed = (
                model.data_preprocessor(
                    data,
                    training=False
                )
            )

            feats = model.extract_feat(
                processed['inputs']
            )

            # ---------------------------------------------
            # 跑一次官方 head，
            # 目的只是让 hook 拿到 CIM 前的 f_kpts
            # ---------------------------------------------
            captured.clear()

            _ = head(feats)

            if 'f_kpts' not in captured:

                raise RuntimeError(
                    'Failed to capture f_kpts '
                    'before CIM.'
                )

            f_kpts = captured['f_kpts']

            # ---------------------------------------------
            # 按原 CIGPose 算 uncertainty Top-1
            # ---------------------------------------------
            initial_x = (
                head.initial_pred_x(
                    f_kpts
                )
            )

            initial_y = (
                head.initial_pred_y(
                    f_kpts
                )
            )

            conf_x = (
                initial_x
                .softmax(dim=-1)
                .max(dim=-1)[0]
            )

            conf_y = (
                initial_y
                .softmax(dim=-1)
                .max(dim=-1)[0]
            )

            confidence = (
                conf_x + conf_y
            ) / 2.0

            uncertainty = (
                1.0 - confidence
            )

            selected_joint = (
                uncertainty
                .argmax(dim=1)
            )

            # ---------------------------------------------
            # 五种 alpha
            # ---------------------------------------------
            predictions = {}

            for alpha in ALPHAS:

                modified = apply_strength(
                    head,
                    f_kpts,
                    selected_joint,
                    alpha
                )

                px, py = (
                    predict_from_joint_features(
                        head,
                        modified
                    )
                )

                predictions[alpha] = (
                    px,
                    py
                )

            data_samples = (
                processed['data_samples']
            )

            batch_n = len(
                data_samples
            )

            for b in range(batch_n):

                if (
                    args.max_persons
                    is not None
                    and global_index
                    >= args.max_persons
                ):
                    break

                info = dataset.get_data_info(
                    global_index
                )

                ann_id = info.get(
                    'id',
                    None
                )

                if ann_id is None:
                    raise RuntimeError(
                        f'No annotation id '
                        f'at index {global_index}'
                    )

                ann_id = int(ann_id)

                raw_ann = anns_by_id[
                    ann_id
                ]

                raw_kpts = np.asarray(
                    raw_ann['keypoints'],
                    dtype=np.float32
                ).reshape(
                    17,
                    3
                )

                gt_xy = raw_kpts[:, :2]

                visibility = (
                    raw_kpts[:, 2]
                    .astype(np.int64)
                )

                labeled_mask = (
                    visibility > 0
                )

                if labeled_mask.sum() == 0:

                    global_index += 1
                    continue

                sample = data_samples[b]
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

                # -----------------------------------------
                # 每个 alpha 的 pose error
                # -----------------------------------------
                errors = {}

                for alpha in ALPHAS:

                    px, py = (
                        predictions[alpha]
                    )

                    pred_xy = (
                        decode_simcc_to_original(
                            px[b],
                            py[b],
                            input_center,
                            input_scale,
                            input_size
                        )
                    )

                    errors[alpha] = pose_error(
                        pred_xy,
                        gt_xy,
                        labeled_mask,
                        raw_ann['bbox']
                    )

                baseline_error = (
                    errors[0.0]
                )

                gains = {}

                for alpha in ALPHAS:

                    gains[alpha] = (
                        baseline_error
                        - errors[alpha]
                    )

                    gains_by_alpha[
                        alpha
                    ].append(
                        gains[alpha]
                    )

                # -----------------------------------------
                # 找这个人的最佳 alpha
                #
                # 因为 SimCC argmax 可能出现并列，
                # 所以这里不强行随便选一个。
                # -----------------------------------------
                gain_array = np.asarray([
                    gains[a]
                    for a in ALPHAS
                ])

                max_gain = float(
                    gain_array.max()
                )

                best_mask = np.isclose(
                    gain_array,
                    max_gain,
                    rtol=0.0,
                    atol=1e-12
                )

                best_indices = np.where(
                    best_mask
                )[0]

                best_alphas = [
                    ALPHAS[i]
                    for i in best_indices
                ]

                # 并列时取平均，
                # 只是为了后面看趋势
                mean_best_alpha = float(
                    np.mean(best_alphas)
                )

                # 每个人在所有并列 best alpha
                # 中平均分一票
                vote = (
                    1.0
                    / len(best_alphas)
                )

                for alpha in best_alphas:

                    best_alpha_soft_counts[
                        alpha
                    ] += vote

                alpha1_best = bool(
                    best_mask[
                        ALPHAS.index(1.0)
                    ]
                )

                alpha1_gain = (
                    gains[1.0]
                )

                strength_oracle_gap = (
                    max_gain
                    - alpha1_gain
                )

                alpha1_is_best.append(
                    int(alpha1_best)
                )

                oracle_minus_alpha1.append(
                    strength_oracle_gap
                )

                joint_id = int(
                    selected_joint[b]
                    .detach()
                    .cpu()
                    .item()
                )

                joint_name = (
                    JOINT_NAMES[joint_id]
                )

                # -----------------------------------------
                # normalized person scale
                # -----------------------------------------
                normalized_scale = (
                    normalized_scale_by_ann[
                        ann_id
                    ]
                )

                scale_group = (
                    get_scale_group(
                        normalized_scale
                    )
                )

                # per-joint stats
                js = (
                    selected_joint_stats[
                        joint_name
                    ]
                )

                js['count'] += 1
                js['best_alpha_sum'] += (
                    mean_best_alpha
                )

                js['alpha1_best'] += int(
                    alpha1_best
                )

                js['oracle_gap_sum'] += (
                    strength_oracle_gap
                )

                # per-scale stats
                ss = (
                    scale_group_stats[
                        scale_group
                    ]
                )

                ss['count'] += 1
                ss['best_alpha_sum'] += (
                    mean_best_alpha
                )

                ss['alpha1_best'] += int(
                    alpha1_best
                )

                ss['oracle_gap_sum'] += (
                    strength_oracle_gap
                )

                row = {
                    'dataset_index':
                        global_index,

                    'ann_id':
                        ann_id,

                    'img_id':
                        int(raw_ann['image_id']),

                    'selected_joint_id':
                        joint_id,

                    'selected_joint_name':
                        joint_name,

                    'normalized_scale':
                        normalized_scale,

                    'scale_group':
                        scale_group,

                    'baseline_error':
                        baseline_error,

                    'best_gain':
                        max_gain,

                    'mean_best_alpha':
                        mean_best_alpha,

                    'alpha1_is_best':
                        int(alpha1_best),

                    'oracle_strength_minus_alpha1':
                        strength_oracle_gap,
                }

                for alpha in ALPHAS:

                    key = str(alpha).replace(
                        '.',
                        '_'
                    )

                    row[
                        f'gain_alpha_{key}'
                    ] = gains[alpha]

                rows.append(row)

                global_index += 1

            if batch_idx % 20 == 0:

                print(
                    f'Processed persons: '
                    f'{global_index}'
                )

    hook_handle.remove()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print()
    print('=' * 82)
    print('RESULT')
    print('=' * 82)

    print(
        f'Persons evaluated = '
        f'{len(rows)}'
    )

    print()
    print(
        'Mean pose-level gain '
        'for each intervention strength:'
    )

    for alpha in ALPHAS:

        values = np.asarray(
            gains_by_alpha[alpha],
            dtype=np.float64
        )

        print(
            f'alpha={alpha:>4.2f}  '
            f'mean gain={values.mean():+.8f}  '
            f'harm rate='
            f'{np.mean(values < 0) * 100:6.2f}%'
        )

    # ---------------------------------------------
    # alpha=1 vs strength oracle
    # ---------------------------------------------
    gaps = np.asarray(
        oracle_minus_alpha1,
        dtype=np.float64
    )

    gap_mean, gap_low, gap_high = (
        mean_ci95(gaps)
    )

    alpha1_best_rate = float(
        np.mean(alpha1_is_best)
    )

    strictly_better_rate = float(
        np.mean(
            gaps > 1e-12
        )
    )

    print()
    print(
        'Original CIGPose alpha=1 '
        'vs best strength:'
    )

    print(
        f'alpha=1 is one of the best '
        f'choices = '
        f'{alpha1_best_rate * 100:.2f}%'
    )

    print(
        f'A different alpha is strictly '
        f'better = '
        f'{strictly_better_rate * 100:.2f}%'
    )

    print(
        f'Mean OracleStrength - alpha1 '
        f'gain = '
        f'{gap_mean:+.8f}'
    )

    print(
        f'95% CI = '
        f'[{gap_low:+.8f}, '
        f'{gap_high:+.8f}]'
    )

    # ---------------------------------------------
    # Best alpha distribution
    # ---------------------------------------------
    print()
    print(
        'Tie-aware best-alpha distribution:'
    )

    n = len(rows)

    for alpha in ALPHAS:

        pct = (
            best_alpha_soft_counts[alpha]
            / n
            * 100
        )

        print(
            f'alpha={alpha:>4.2f}: '
            f'{pct:6.2f}%'
        )

    # ---------------------------------------------
    # By scale
    # ---------------------------------------------
    print()
    print('=' * 82)
    print('BY NORMALIZED PERSON SCALE')
    print('=' * 82)

    print(
        f'{"group":10s}'
        f' {"N":>7s}'
        f' {"mean_best_alpha":>18s}'
        f' {"alpha1_best":>14s}'
        f' {"oracle_gap":>14s}'
    )

    for group in [
        'small',
        'medium',
        'large'
    ]:

        s = scale_group_stats[
            group
        ]

        if s['count'] == 0:
            continue

        count = s['count']

        print(
            f'{group:10s}'
            f' {count:7d}'
            f' {s["best_alpha_sum"] / count:18.3f}'
            f' {s["alpha1_best"] / count * 100:13.2f}%'
            f' {s["oracle_gap_sum"] / count:+14.8f}'
        )

    # ---------------------------------------------
    # By selected joint
    # ---------------------------------------------
    print()
    print('=' * 82)
    print('BY OFFICIAL SELECTED JOINT')
    print('=' * 82)

    print(
        f'{"joint":16s}'
        f' {"N":>6s}'
        f' {"mean_best_a":>12s}'
        f' {"a1_best":>10s}'
        f' {"oracle_gap":>14s}'
    )

    for joint_name in JOINT_NAMES:

        s = selected_joint_stats[
            joint_name
        ]

        if s['count'] == 0:
            continue

        count = s['count']

        print(
            f'{joint_name:16s}'
            f' {count:6d}'
            f' {s["best_alpha_sum"] / count:12.3f}'
            f' {s["alpha1_best"] / count * 100:9.2f}%'
            f' {s["oracle_gap_sum"] / count:+14.8f}'
        )

    # ---------------------------------------------------------
    # Final interpretation
    # ---------------------------------------------------------
    print()
    print('=' * 82)
    print('INTERPRETATION')
    print('=' * 82)

    if (
        gap_low > 0
        and strictly_better_rate > 0.20
    ):

        print(
            'SUPPORTED: fixing the selected joint '
            'with a single fixed alpha=1 is not '
            'optimal for many persons.'
        )

        print(
            'This supports adaptive refinement '
            'strength after joint selection.'
        )

    elif gap_low > 0:

        print(
            'PARTIAL SUPPORT: adaptive strength '
            'has a statistically positive advantage, '
            'but the practical effect may be limited.'
        )

    else:

        print(
            'NOT SUPPORTED: current experiment '
            'does not show a clear advantage over '
            'the fixed alpha=1 refinement.'
        )

    # ---------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------
    csv_path = os.path.join(
        args.output_dir,
        'challenge2_intervention_strength.csv'
    )

    if rows:

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
        'Saved:',
        csv_path
    )


if __name__ == '__main__':
    main()
