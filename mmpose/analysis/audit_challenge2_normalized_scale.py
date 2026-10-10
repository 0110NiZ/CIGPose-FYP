import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal


CSV_PATH = 'analysis/results/challenge1_coco_utility_gap.csv'
ANN_PATH = '../data/coco/annotations/person_keypoints_val2017.json'

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
    """Benjamini-Hochberg 多重检验修正。"""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted = ranked * n / np.arange(1, n + 1)

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(n)
    result[order] = adjusted

    return result


def main():

    print('=' * 82)
    print(
        'Challenge 2 Diagnostic - Normalized Person Scale '
        'vs Refinement Utility'
    )
    print('=' * 82)

    # ---------------------------------------------------------
    # 1. 读取之前已经算好的 17-joint intervention gain
    # ---------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    print('Persons in utility CSV:', len(df))

    # ---------------------------------------------------------
    # 2. 读取 COCO annotation 和图像尺寸
    # ---------------------------------------------------------
    with open(
        ANN_PATH,
        'r',
        encoding='utf-8'
    ) as f:
        coco = json.load(f)

    anns = {
        int(ann['id']): ann
        for ann in coco['annotations']
    }

    images = {
        int(img['id']): img
        for img in coco['images']
    }

    # ---------------------------------------------------------
    # 3. 计算 normalized person scale
    #
    # 原来：
    # sqrt(bbox_w * bbox_h)
    #
    # 现在：
    # sqrt(
    #   (bbox_w / image_w)
    #   *
    #   (bbox_h / image_h)
    # )
    #
    # 人话：
    # 这个人的 bbox 占整张图的比例有多大。
    # ---------------------------------------------------------
    normalized_scales = []

    for ann_id in df['ann_id']:

        ann = anns[int(ann_id)]

        img = images[int(ann['image_id'])]

        _, _, bbox_w, bbox_h = ann['bbox']

        image_w = float(img['width'])
        image_h = float(img['height'])

        relative_w = float(bbox_w) / image_w
        relative_h = float(bbox_h) / image_h

        normalized_scale = np.sqrt(
            max(relative_w * relative_h, 1e-12)
        )

        normalized_scales.append(
            normalized_scale
        )

    df['normalized_scale'] = normalized_scales

    # ---------------------------------------------------------
    # 4. 仍然按人数三等分
    #
    # small / medium / large 各约 1/3
    # ---------------------------------------------------------
    q1 = df['normalized_scale'].quantile(1 / 3)
    q2 = df['normalized_scale'].quantile(2 / 3)

    def scale_group(x):

        if x <= q1:
            return 'small'

        elif x <= q2:
            return 'medium'

        else:
            return 'large'

    df['scale_group'] = (
        df['normalized_scale']
        .apply(scale_group)
    )

    print()
    print('Normalized scale thresholds:')

    print(
        f'small  <= {q1:.6f}'
    )

    print(
        f'medium <= {q2:.6f}'
    )

    print(
        f'large  >  {q2:.6f}'
    )

    print()
    print('Persons per group:')

    print(
        df['scale_group'].value_counts()
    )

    # ---------------------------------------------------------
    # 5. 每一个 joint 单独分析
    #
    # 重点还是：
    # 同一个 joint 在不同 person scale 下比较。
    # ---------------------------------------------------------
    results = []

    for joint_id, joint_name in enumerate(
        JOINT_NAMES
    ):

        gain_col = (
            f'gain_{joint_id}_{joint_name}'
        )

        gain = df[gain_col].to_numpy(
            dtype=np.float64
        )

        scale = df[
            'normalized_scale'
        ].to_numpy(
            dtype=np.float64
        )

        small = df.loc[
            df['scale_group'] == 'small',
            gain_col
        ].to_numpy(
            dtype=np.float64
        )

        medium = df.loc[
            df['scale_group'] == 'medium',
            gain_col
        ].to_numpy(
            dtype=np.float64
        )

        large = df.loc[
            df['scale_group'] == 'large',
            gain_col
        ].to_numpy(
            dtype=np.float64
        )

        # -----------------------------
        # 平均 refinement gain
        # -----------------------------
        mean_small = float(
            np.mean(small)
        )

        mean_medium = float(
            np.mean(medium)
        )

        mean_large = float(
            np.mean(large)
        )

        gain_means = np.asarray([
            mean_small,
            mean_medium,
            mean_large,
        ])

        gain_range = float(
            gain_means.max()
            - gain_means.min()
        )

        # -----------------------------
        # Harm rate
        # gain < 0 = 修完更差
        # -----------------------------
        harm_small = float(
            np.mean(small < 0)
        )

        harm_medium = float(
            np.mean(medium < 0)
        )

        harm_large = float(
            np.mean(large < 0)
        )

        harm_rates = np.asarray([
            harm_small,
            harm_medium,
            harm_large,
        ])

        harm_range = float(
            harm_rates.max()
            - harm_rates.min()
        )

        # -----------------------------
        # 连续 normalized scale
        # 和 gain 的 Spearman 相关性
        # -----------------------------
        rho, rho_p = spearmanr(
            scale,
            gain
        )

        # -----------------------------
        # small / medium / large
        # 三组 gain 是否存在统计差异
        # -----------------------------
        _, kw_p = kruskal(
            small,
            medium,
            large
        )

        results.append({
            'joint_id':
                joint_id,

            'joint':
                joint_name,

            'small_gain':
                mean_small,

            'medium_gain':
                mean_medium,

            'large_gain':
                mean_large,

            'gain_range':
                gain_range,

            'small_harm_rate':
                harm_small,

            'medium_harm_rate':
                harm_medium,

            'large_harm_rate':
                harm_large,

            'harm_range':
                harm_range,

            'spearman_rho':
                float(rho),

            'spearman_p':
                float(rho_p),

            'kruskal_p':
                float(kw_p),
        })

    # ---------------------------------------------------------
    # 6. 对 17 个 joint 做 FDR correction
    # ---------------------------------------------------------
    q_values = bh_fdr([
        r['kruskal_p']
        for r in results
    ])

    for result, q_value in zip(
        results,
        q_values
    ):

        result['q_value'] = float(
            q_value
        )

    # 按 q-value 排序
    results.sort(
        key=lambda x: x['q_value']
    )

    # ---------------------------------------------------------
    # 7. Gain 结果
    # ---------------------------------------------------------
    print()
    print('=' * 82)
    print('PER-JOINT RESULT')
    print('=' * 82)

    header = (
        f'{"joint":16s}'
        f' {"small":>10s}'
        f' {"medium":>10s}'
        f' {"large":>10s}'
        f' {"range":>10s}'
        f' {"rho":>8s}'
        f' {"q":>11s}'
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
            f' {r["q_value"]:11.3e}'
        )

    # ---------------------------------------------------------
    # 8. Harm rate
    # ---------------------------------------------------------
    print()
    print('=' * 82)
    print('HARM RATE BY NORMALIZED SCALE')
    print('=' * 82)

    header = (
        f'{"joint":16s}'
        f' {"small":>10s}'
        f' {"medium":>10s}'
        f' {"large":>10s}'
        f' {"range":>10s}'
    )

    print(header)
    print('-' * len(header))

    for r in results:

        print(
            f'{r["joint"]:16s}'
            f' {r["small_harm_rate"] * 100:9.2f}%'
            f' {r["medium_harm_rate"] * 100:9.2f}%'
            f' {r["large_harm_rate"] * 100:9.2f}%'
            f' {r["harm_range"] * 100:9.2f}%'
        )

    # ---------------------------------------------------------
    # 9. Summary
    #
    # 注意：
    # q < .05 只能说明统计上有差异。
    # 样本很多时非常小的差异也会显著。
    #
    # 所以另外找：
    # |rho| >= 0.10
    # 作为“实际关系比较值得关注”的简单参考。
    # ---------------------------------------------------------
    significant = [
        r for r in results
        if r['q_value'] < 0.05
    ]

    meaningful = [
        r for r in results
        if (
            r['q_value'] < 0.05
            and abs(r['spearman_rho']) >= 0.10
        )
    ]

    print()
    print('=' * 82)
    print('SUMMARY')
    print('=' * 82)

    print(
        'Statistically significant joints '
        f'(q < 0.05): '
        f'{len(significant)} / 17'
    )

    print(
        'Stronger scale-related joints '
        f'(q < 0.05 and |rho| >= 0.10): '
        f'{len(meaningful)} / 17'
    )

    if meaningful:

        print()
        print('Stronger scale-related joints:')

        for r in meaningful:

            print(
                f'  {r["joint"]:16s}'
                f' gain_range={r["gain_range"]:.6f}'
                f' harm_range='
                f'{r["harm_range"] * 100:.2f}pp'
                f' rho={r["spearman_rho"]:+.3f}'
                f' q={r["q_value"]:.3e}'
            )

    print()

    if len(meaningful) >= 4:

        print(
            'SUPPORTED: normalized person scale '
            'still shows a meaningful relationship '
            'with refinement utility for multiple joints.'
        )

    elif len(meaningful) > 0:

        print(
            'PARTIAL SUPPORT: normalized scale '
            'affects some joints, but the effect '
            'is not widespread.'
        )

    else:

        print(
            'NOT SUPPORTED: after normalizing by '
            'image size, scale dependence becomes weak.'
        )


if __name__ == '__main__':
    main()
