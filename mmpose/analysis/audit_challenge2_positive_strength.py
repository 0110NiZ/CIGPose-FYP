import math
import numpy as np
import pandas as pd


CSV_PATH = 'analysis/results/challenge2_intervention_strength.csv'

ALPHAS = [0.25, 0.50, 0.75, 1.00]

COLS = {
    0.25: 'gain_alpha_0_25',
    0.50: 'gain_alpha_0_5',
    0.75: 'gain_alpha_0_75',
    1.00: 'gain_alpha_1_0',
}


def mean_ci95(x):
    x = np.asarray(x, dtype=np.float64)

    mean = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))

    return (
        mean,
        mean - 1.96 * se,
        mean + 1.96 * se
    )


df = pd.read_csv(CSV_PATH)

print('=' * 76)
print('Challenge 2 - Positive Refinement Strength Audit')
print('Assume the selected joint WILL be refined.')
print('=' * 76)

print('Persons:', len(df))

gains = np.stack(
    [df[COLS[a]].to_numpy() for a in ALPHAS],
    axis=1
)

# 固定 CIGPose alpha=1
alpha1_gain = gains[:, -1]

# 只在正强度 0.25/0.5/0.75/1 中找最佳
best_gain = gains.max(axis=1)

gap = best_gain - alpha1_gain

# alpha=1 是否属于最佳之一
alpha1_best = np.isclose(
    alpha1_gain,
    best_gain,
    atol=1e-12,
    rtol=0.0
)

# 是否有其他正强度严格比 alpha=1 好
different_better = gap > 1e-12

mean_gap, low_gap, high_gap = mean_ci95(gap)

print()
print('Fixed alpha=1 vs best positive strength:')
print(
    f'alpha=1 is one of the best choices = '
    f'{alpha1_best.mean() * 100:.2f}%'
)

print(
    f'Another positive alpha is strictly better = '
    f'{different_better.mean() * 100:.2f}%'
)

print(
    f'Mean BestPositive - alpha1 gain = '
    f'{mean_gap:+.8f}'
)

print(
    f'95% CI = [{low_gap:+.8f}, {high_gap:+.8f}]'
)

# ---------------------------------------------------------
# 最佳正强度分布，处理并列
# ---------------------------------------------------------
counts = {a: 0.0 for a in ALPHAS}

for row in gains:
    m = row.max()

    mask = np.isclose(
        row,
        m,
        atol=1e-12,
        rtol=0.0
    )

    idxs = np.where(mask)[0]

    vote = 1.0 / len(idxs)

    for idx in idxs:
        counts[ALPHAS[idx]] += vote

print()
print('Tie-aware best positive-alpha distribution:')

for a in ALPHAS:
    print(
        f'alpha={a:>4.2f}: '
        f'{counts[a] / len(df) * 100:6.2f}%'
    )

# ---------------------------------------------------------
# 各 alpha 自己的平均 gain / harm rate
# ---------------------------------------------------------
print()
print('Mean gain and harm rate:')

for i, a in enumerate(ALPHAS):
    x = gains[:, i]

    print(
        f'alpha={a:>4.2f}  '
        f'mean gain={x.mean():+.8f}  '
        f'harm rate={(x < 0).mean() * 100:6.2f}%'
    )

print()
print('=' * 76)
print('INTERPRETATION')
print('=' * 76)

if low_gap > 0 and different_better.mean() > 0.20:
    print(
        'SUPPORTED: even when refinement is required, '
        'a fixed alpha=1 is not optimal for many persons.'
    )
    print(
        'This directly supports adaptive refinement strength.'
    )
elif low_gap > 0:
    print(
        'PARTIAL SUPPORT: adaptive positive strength has '
        'a significant but limited advantage.'
    )
else:
    print(
        'NOT SUPPORTED: positive-strength adaptation does '
        'not clearly improve over fixed alpha=1.'
    )
