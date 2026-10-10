import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

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
        default=None,
        help='只跑前 N 个 person；不填就是完整 COCO val'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--output-dir',
        default='analysis/results'
    )

    return parser.parse_args()


def unwrap_dataset_cfg(dataset_cfg):
    """找到真正带 ann_file 的 dataset config。"""
    current = dataset_cfg

    while 'ann_file' not in current and 'dataset' in current:
        current = current['dataset']

    return current


def mean_ci95(values):
    """均值的简单 95% CI。"""
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')

    mean = float(values.mean())

    if len(values) == 1:
        return mean, float('nan'), float('nan')

    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    low = mean - 1.96 * se
    high = mean + 1.96 * se

    return mean, low, high


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('Challenge 1 - Step 3')
    print('COCO uncertainty ranking diagnostic')
    print('=' * 70)

    # ---------------------------------------------------------
    # 1. Load config
    # ---------------------------------------------------------
    cfg = Config.fromfile(args.config)

    cfg.load_from = None
    cfg.work_dir = 'work_dirs/challenge1_coco_uncertainty_audit'

    # 为了稳定、容易 debug
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    # ---------------------------------------------------------
    # 2. 强制使用 GT bbox
    #
    # 我们现在不是测 AP，而是在研究：
    # “这个 GT person 的 Top-1 uncertainty joint 是不是遮挡”
    #
    # 所以用 GT person 最干净。
    # ---------------------------------------------------------
    test_dataset_cfg = unwrap_dataset_cfg(cfg.test_dataloader.dataset)

    if 'bbox_file' in test_dataset_cfg:
        print('Original bbox_file:', test_dataset_cfg.get('bbox_file'))
        test_dataset_cfg['bbox_file'] = None
        print('Diagnostic bbox_file: None (use GT person bbox)')

    # 找 COCO annotation json
    data_root = test_dataset_cfg.get('data_root', '')
    ann_file = test_dataset_cfg['ann_file']

    if os.path.isabs(ann_file):
        ann_path = ann_file
    else:
        ann_path = os.path.join(data_root, ann_file)

    ann_path = os.path.normpath(ann_path)

    print('Annotation file:', ann_path)

    # ---------------------------------------------------------
    # 3. 读原始 COCO annotation
    #    这里必须读原始 keypoints 的第三维 v
    # ---------------------------------------------------------
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    anns_by_id = {
        int(ann['id']): ann
        for ann in coco['annotations']
    }

    anns_by_img = defaultdict(list)

    for ann in coco['annotations']:
        if ann.get('category_id') == 1:
            anns_by_img[int(ann['image_id'])].append(ann)

    # ---------------------------------------------------------
    # 4. Build runner / model
    # ---------------------------------------------------------
    runner = Runner.from_cfg(cfg)

    # EMAHook 需要先初始化，再加载 checkpoint
    runner.call_hook('before_run')

    print()
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
    # 5. Storage
    # ---------------------------------------------------------
    rows = []

    top1_joint_counter = Counter()

    official_hits = []
    random_probs = []

    official_hits_eligible = []
    random_probs_eligible = []

    global_index = 0

    # ---------------------------------------------------------
    # 6. Iterate COCO val persons
    # ---------------------------------------------------------
    with torch.no_grad():

        for batch_idx, data in enumerate(dataloader):

            if args.max_persons is not None:
                if global_index >= args.max_persons:
                    break

            processed = model.data_preprocessor(
                data,
                training=False
            )

            feats = model.extract_feat(processed['inputs'])

            # 我们之前加进 CIGHead 的 diagnostic_forward
            diag = model.head.diagnostic_forward(feats)

            uncertainty = (
                diag['confound_score']
                .detach()
                .cpu()
                .numpy()
            )

            top1_indices = (
                diag['top1_indices']
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            data_samples = processed['data_samples']

            batch_size = len(data_samples)

            for b in range(batch_size):

                if args.max_persons is not None:
                    if global_index >= args.max_persons:
                        break

                sample = data_samples[b]

                # -------------------------------------------------
                # 找这个 person 对应的 COCO annotation id
                # -------------------------------------------------
                data_info = dataset.get_data_info(global_index)

                ann_id = data_info.get('id', None)

                raw_ann = None

                if ann_id is not None:
                    ann_id = int(ann_id)
                    raw_ann = anns_by_id.get(ann_id, None)

                # 正常 GT bbox 模式应该直接能找到
                if raw_ann is None:
                    raise RuntimeError(
                        f'Cannot find raw COCO annotation for '
                        f'dataset index {global_index}, '
                        f'ann_id={ann_id}.'
                    )

                # -------------------------------------------------
                # COCO keypoints:
                # [x, y, v, x, y, v, ...]
                # -------------------------------------------------
                raw_kpts = np.asarray(
                    raw_ann['keypoints'],
                    dtype=np.float32
                ).reshape(-1, 3)

                if raw_kpts.shape[0] != 17:
                    raise RuntimeError(
                        f'Expected 17 COCO joints, got '
                        f'{raw_kpts.shape[0]}'
                    )

                visibility = raw_kpts[:, 2].astype(np.int64)

                # v == 1 = labelled but not visible / occluded
                occluded = (visibility == 1)

                n_occluded = int(occluded.sum())

                # 有标签关节：
                # v=1 = 有标注但不可见/遮挡
                # v=2 = 有标注且可见
                labeled = (visibility == 1) | (visibility == 2)
                n_labeled = int(labeled.sum())

                # -------------------------------------------------
                # Official CIGPose uncertainty Top-1
                # -------------------------------------------------
                top1_joint = int(top1_indices[b])

                official_hit = int(
                    visibility[top1_joint] == 1
                )

                # -------------------------------------------------
                # Random baseline
                #
                # 和 OCHuman diagnostic 保持完全一致：
                # 只在有标签关节 (v=1 或 v=2) 中随机选择。
                #
                # P(random hit)
                #   = #v1 / (#v1 + #v2)
                #
                # v=0 不加入随机池，因为这些关节没有有效标注，
                # 无法判断是“遮挡”还是“可见”。
                # -------------------------------------------------
                if n_labeled > 0:
                    random_prob = n_occluded / n_labeled
                else:
                    random_prob = float('nan')

                official_hits.append(official_hit)
                random_probs.append(random_prob)

                # 如果这个 person 至少存在一个遮挡 joint，
                # 再额外算一份 conditional result
                if n_occluded > 0:
                    official_hits_eligible.append(
                        official_hit
                    )
                    random_probs_eligible.append(
                        random_prob
                    )

                top1_joint_counter[
                    JOINT_NAMES[top1_joint]
                ] += 1

                # -------------------------------------------------
                # Top1 / Top2 uncertainty gap
                # 顺便记录，后面可能会用
                # -------------------------------------------------
                u = uncertainty[b]

                order = np.argsort(-u)

                top1_u = float(u[order[0]])
                top2_u = float(u[order[1]])
                u_gap = top1_u - top2_u

                rows.append({
                    'dataset_index': global_index,
                    'ann_id': int(raw_ann['id']),
                    'img_id': int(raw_ann['image_id']),

                    'top1_joint_id': top1_joint,
                    'top1_joint_name':
                        JOINT_NAMES[top1_joint],

                    'top1_uncertainty': top1_u,
                    'top2_uncertainty': top2_u,
                    'top1_top2_gap': u_gap,

                    'top1_visibility':
                        int(visibility[top1_joint]),

                    'top1_is_occluded':
                        official_hit,

                    'num_occluded_joints':
                        n_occluded,

                    'num_labeled_joints':
                        n_labeled,

                    'random_occlusion_probability':
                        random_prob,

                    'has_occluded_joint':
                        int(n_occluded > 0),
                })

                global_index += 1

            if batch_idx % 20 == 0:
                print(
                    f'Processed persons: {global_index}'
                )

    # ---------------------------------------------------------
    # 7. Calculate statistics
    #
    # 主指标与 OCHuman diagnostic 完全一致：
    # 分母只包括至少有一个 v=1 遮挡关节的 person。
    # ---------------------------------------------------------
    if len(official_hits_eligible) == 0:
        raise RuntimeError(
            'No person with at least one v=1 occluded joint was found.'
        )

    official_rate = float(
        np.mean(official_hits_eligible)
    )

    random_rate = float(
        np.mean(random_probs_eligible)
    )

    difference = official_rate - random_rate

    ratio = (
        official_rate / random_rate
        if random_rate > 0
        else float('nan')
    )

    # paired difference:
    # 对同一个 person：
    # Official 是否命中(0/1)
    # 减去
    # 在该 person 的有标签关节中随机命中遮挡的概率
    paired_diff = (
        np.asarray(official_hits_eligible, dtype=np.float64)
        -
        np.asarray(random_probs_eligible, dtype=np.float64)
    )

    diff_mean, diff_low, diff_high = mean_ci95(
        paired_diff
    )

    # ---------------------------------------------------------
    # 8. Decide whether ranking shows useful signal
    # ---------------------------------------------------------
    ranking_supported = (
        difference > 0
        and not math.isnan(diff_low)
        and diff_low > 0
    )

    # ---------------------------------------------------------
    # 9. Print
    # ---------------------------------------------------------
    print()
    print('=' * 70)
    print('RESULT')
    print('=' * 70)

    print(f'Total COCO persons evaluated: {len(rows)}')
    print(
        f'Persons with >=1 v=1 joint: '
        f'{len(official_hits_eligible)}'
    )

    print()
    print('Main diagnostic (same definition as OCHuman):')

    print(
        f'Official Top-1 occlusion hit rate = '
        f'{official_rate * 100:.2f}%'
    )

    print(
        f'Random labeled-joint hit rate     = '
        f'{random_rate * 100:.2f}%'
    )

    print(
        f'Difference                        = '
        f'{difference * 100:+.2f} percentage points'
    )

    print(
        f'Official / Random                 = '
        f'{ratio:.3f}x'
    )

    print()
    print('Paired difference 95% CI:')
    print(
        f'{diff_mean * 100:+.2f} pp '
        f'[{diff_low * 100:+.2f}, '
        f'{diff_high * 100:+.2f}]'
    )

    print()
    print('Top-1 selected joint frequency:')

    for name, count in top1_joint_counter.most_common():
        print(
            f'{name:16s}: '
            f'{count:5d} / {len(rows)}'
        )

    print()
    if ranking_supported:
        print(
            'PASS: Official uncertainty Top-1 is '
            'better than random at selecting '
            'occluded joints on COCO.'
        )
    else:
        print(
            'NOT YET SUPPORTED: Current result does '
            'not provide clear evidence that the '
            'Top-1 uncertainty ranking beats random.'
        )

    # ---------------------------------------------------------
    # 10. Save CSV
    # ---------------------------------------------------------
    csv_path = os.path.join(
        args.output_dir,
        'challenge1_step3_coco_uncertainty_ranking.csv'
    )

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

    # ---------------------------------------------------------
    # 11. Save summary JSON
    #
    # 同时记录：
    # 做了什么 / 为什么做 / 结果 / 是否有效
    # ---------------------------------------------------------
    summary = {
        'step': 'Challenge 1 - Step 3',

        'what_we_did':
            'Evaluate whether CIGPose uncertainty '
            'Top-1 selection is more likely than '
            'random selection to choose an occluded '
            'COCO joint.',

        'why_we_did_it':
            'The initial prediction heads were found '
            'to receive no gradient in the released '
            'training path, but this alone does not '
            'prove that their ranking is useless. '
            'This experiment directly tests whether '
            'the resulting ranking contains useful '
            'occlusion-related information.',

        'occlusion_definition':
            'COCO keypoint visibility v == 1',

        'random_definition':
            'For each person with at least one '
            'occluded joint, randomly select among '
            'labeled joints only (v=1 or v=2): '
            '#v1 / (#v1 + #v2).',

        'num_persons': len(rows),

        'overall': {
            'official_top1_hit_rate':
                official_rate,

            'random_expected_hit_rate':
                random_rate,

            'difference_percentage_points':
                difference * 100,

            'official_over_random_ratio':
                ratio,

            'paired_difference_95ci_pp': [
                diff_low * 100,
                diff_high * 100,
            ],
        },

        'persons_with_occlusion': {
            'num_persons':
                len(official_hits_eligible),

            'official_top1_hit_rate':
                official_rate,

            'random_labeled_joint_hit_rate':
                random_rate,
        },

        'top1_joint_frequency':
            dict(top1_joint_counter),

        'is_effective':
            bool(ranking_supported),

        'interpretation':
            (
                'The uncertainty ranking shows '
                'evidence of useful occlusion-related '
                'selection ability on COCO.'
                if ranking_supported
                else
                'The current experiment does not yet '
                'show clear evidence that uncertainty '
                'Top-1 beats random selection.'
            ),

        'next_step':
            (
                'Run the Utility Gap Audit: compare '
                'Random Top-1, Official uncertainty '
                'Top-1, and Oracle utility Top-1.'
                if ranking_supported
                else
                'Inspect the ranking definition and '
                'additional difficulty/error metrics '
                'before proceeding to utility '
                'prediction.'
            ),
    }

    json_path = os.path.join(
        args.output_dir,
        'challenge1_step3_coco_uncertainty_ranking_summary.json'
    )

    with open(
        json_path,
        'w',
        encoding='utf-8'
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print('Saved:')
    print(csv_path)
    print(json_path)


if __name__ == '__main__':
    main()
