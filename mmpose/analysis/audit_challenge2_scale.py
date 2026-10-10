import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal


CSV_PATH = 'analysis/results/challenge1_coco_utility_gap.csv'

CONFIG_ANN_PATH = '../data/coco/annotations/person_keypoints_val2017.json'

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


def bh_fdr(p_values):
    """
    Benjamini-Hochberg FDR correction.
    17 个 joint 一起检验，避免“碰巧显著”。
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted = ranked * n / np.arange(1, n + 1)

    # 保证单调
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(n)
    result[order] = adjusted

    return result


def main():

    print('=' * 78)
    print('Challenge 2 Diagnostic - Person Scale vs Refinement Utility')
    print('=' * 78)

    # ---------------------------------------------------------
    # 1. 读刚才 Challenge 1 生成的真实 intervention gain
    # ---------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    print('Persons in utility CSV:', len(df))

    # ---------------------------------------------------------
    # 2. 读 COCO 原始 bbox
    # ---------------------------------------------------------
    with open(
        CONFIG_ANN_PATH,
        'r',
        encoding='utf-8'
    ) as f:
        coco = json.load(f)

    anns = {
        int(ann['id']): ann
        for ann in coco['annotations']
    }

    # ---------------------------------------------------------
    # 3. 给每个人计算 scale
    #
    # bbox 是 [x, y, w, h]
    #
    # scale = sqrt(w*h)
    #
    # 人话：
    # bbox 越大，这个人在原图中占的像素越多。
    # ---------------------------------------------------------
    scales = []

    for ann_id in df['ann_id']:

        ann = anns[int(ann_id)]

        _, _, w, h = ann['bbox']

        scale = np.sqrt(
            max(float(w) * float(h), 1e-12)
        )

        scales.append(scale)

    df['person_scale'] = scales

    # ---------------------------------------------------------
    # 4. 按 scale 三等分
    #
    # 不直接用固定像素阈值，
    # 先保证 small / medium / large 人数基本一致。
    # ---------------------------------------------------------
    q1 = df['person_scale'].quantile(1 / 3)
    q2 = df['person_scale'].quantile(2 / 3)

    def scale_group(x):
        if x <= q1:
            return 'small'
        elif x <= q2:
            return 'medium'
        else:
            return 'large'

    df['scale_group'] = (
        df['person_scale']
        .apply(scale_group)
    )

    print()
    print('Scale thresholds:')
    print(f'small  <= {q1:.2f}')
    print(f'medium <= {q2:.2f}')
    print(f'large  >  {q2:.2f}')

    print()
    print('Persons per group:')
    print(df['scale_group'].value_counts())

    # ---------------------------------------------------------
    # 5. 对每一个 joint 单独分析
    #
    # 重点：
    # right_wrist 只和 right_wrist 自己比较
    # ankle 只和 ankle 自己比较
    #
    # 不是拿不同 joint 互相比较。
    # ---------------------------------------------------------
    results = []

    for j, joint_name in enumerate(JOINT_NAMES):

        gain_col = f'gain_{j}_{joint_name}'

        gain = df[gain_col].to_numpy()
        scale = df['person_scale'].to_numpy()

        small = df.loc[
            df['scale_group'] == 'small',
            gain_col
        ].to_numpy()

        medium = df.loc[
            df['scale_group'] == 'medium',
            gain_col
        ].to_numpy()

        large = df.loc[
            df['scale_group'] == 'large',
            gain_col
        ].to_numpy()

        # 各尺度平均收益
        mean_small = float(np.mean(small))
        mean_medium = float(np.mean(medium))
        mean_large = float(np.mean(large))

        # harm rate:
        # gain < 0 表示修完反而更差
        harm_small = float(np.mean(small < 0))
        harm_medium = float(np.mean(medium < 0))
        harm_large = float(np.mean(large < 0))

        # 连续 scale 和 gain 的关系
        rho, rho_p = spearmanr(
            scale,
            gain
        )

        # 三个 scale group 的 gain 分布是否存在差异
        kw_stat, kw_p = kruskal(
            small,
            medium,
            large
        )

        # 最大组间平均 gain 差距
        means = np.array([
            mean_small,
            mean_medium,
            mean_large,
        ])

        mean_gain_range = float(
            means.max() - means.min()
        )

        results.append({
            'joint_id': j,
            'joint': joint_name,

            'small_gain':
                mean_small,

            'medium_gain':
                mean_medium,

            'large_gain':
                mean_large,

            'gain_range':
                mean_gain_range,

            'small_harm_rate':
                harm_small,

            'medium_harm_rate':
                harm_medium,

            'large_harm_rate':
                harm_large,

            'spearman_rho':
                float(rho),

            'spearman_p':
                float(rho_p),

            'kruskal_p':
                float(kw_p),
        })

    # ---------------------------------------------------------
    # 6. 多重检验修正
    # ---------------------------------------------------------
    kw_ps = [
        r['kruskal_p']
        for r in results
    ]

    q_values = bh_fdr(kw_ps)

    for r, q in zip(results, q_values):
        r['q_value'] = float(q)

    # 按 q-value 排
    results.sort(
        key=lambda x: x['q_value']
    )

    # ---------------------------------------------------------
    # 7. 输出
    # ---------------------------------------------------------
    print()
    print('=' * 78)
    print('PER-JOINT RESULT')
    print('=' * 78)

    header = (
        f'{"joint":16s}'
        f' {"small":>10s}'
        f' {"medium":>10s}'
        f' {"large":>10s}'
        f' {"range":>10s}'
        f' {"rho":>8s}'
        f' {"q":>10s}'
    )

    print(header)
    print('-' * len(header))

    for r in results:

        print(
            f'{r["joint"]:16s}'
            f' {r["small_gain"]:+10.6f}'
            f' {r["medium_gain"]:+10.6f}'
            f' {r["large_gain"]:+10.6f}'
            f' {r["gain_range"]:10.6f}'
            f' {r["spearman_rho"]:+8.3f}'
            f' {r["q_value"]:10.3e}'
        )

    # ---------------------------------------------------------
    # 8. Harm rate
    # ---------------------------------------------------------
    print()
    print('=' * 78)
    print('HARM RATE BY SCALE')
    print('=' * 78)

    print(
        f'{"joint":16s}'
        f' {"small":>10s}'
        f' {"medium":>10s}'
        f' {"large":>10s}'
    )

    for r in results:

        print(
            f'{r["joint"]:16s}'
            f' {r["small_harm_rate"] * 100:9.2f}%'
            f' {r["medium_harm_rate"] * 100:9.2f}%'
            f' {r["large_harm_rate"] * 100:9.2f}%'
        )

    # ---------------------------------------------------------
    # 9. 总结
    # ---------------------------------------------------------
    significant = [
        r for r in results
        if r['q_value'] < 0.05
    ]

    print()
    print('=' * 78)
    print('SUMMARY')
    print('=' * 78)

    print(
        'Joints with significant scale-dependent '
        f'gain after FDR correction: '
        f'{len(significant)} / 17'
    )

    if significant:

        print()
        print('Significant joints:')

        for r in significant:
            print(
                f'  {r["joint"]:16s}'
                f' gain range={r["gain_range"]:.6f}'
                f' rho={r["spearman_rho"]:+.3f}'
                f' q={r["q_value"]:.3e}'
            )

    print()

    if len(significant) >= 4:

        print(
            'SUPPORTED: refinement utility shows '
            'clear dependence on person scale '
            'for multiple joints.'
        )

    elif len(significant) > 0:

        print(
            'PARTIAL SUPPORT: scale affects some '
            'joints, but the effect is not widespread.'
        )

    else:

        print(
            'NOT SUPPORTED: current data does not '
            'show clear scale dependence.'
        )


if __name__ == '__main__':
    main()
