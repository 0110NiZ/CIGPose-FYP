import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ============================================================
# Paths
# ============================================================
base_dir = "analysis/final_outputs"
out_dir = os.path.join(base_dir, "figures_v5")
os.makedirs(out_dir, exist_ok=True)

utility_csv = os.path.join(
    base_dir,
    "challenge1_coco_utility_gap.csv"
)

strength_csv = os.path.join(
    base_dir,
    "challenge2_intervention_strength.csv"
)

df1 = pd.read_csv(utility_csv)
df2 = pd.read_csv(strength_csv)

N = len(df1)

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,

    "axes.titlesize": 11.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 10.5,

    "xtick.labelsize": 9.3,
    "ytick.labelsize": 9.3,

    "axes.linewidth": 0.8,
    "axes.edgecolor": "#777777",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "axes.unicode_minus": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# Palette
# ============================================================
SOFT_BLUE = "#D3DCF0"
MID_BLUE = "#5271B6"

SOFT_PINK = "#F2B7BE"
MID_PINK = "#DE837D"

ZERO_YELLOW = "#F6E295"

GRID = "#E8EBF0"
TEXT = "#333333"
MUTED = "#666666"
WHITE = "#FFFFFF"

diff_cmap = LinearSegmentedColormap.from_list(
    "difference_map",
    [
        MID_BLUE,
        "#DDE7F5",
        WHITE,
        "#F7DADD",
        MID_PINK,
    ]
)

# ============================================================
# Helpers
# ============================================================
def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")

    ax.tick_params(
        axis="both",
        width=0.7,
        length=3.5,
        color="#777777",
        labelcolor=TEXT
    )

    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            color=GRID,
            linewidth=0.75,
            alpha=1.0
        )

    ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT
    )

# ============================================================
# FIGURE 3.1 DATA
# ============================================================
mean_random = df1[
    "random_expected_gain"
].mean()

mean_uncertainty = df1[
    "official_gain"
].mean()

mean_oracle = df1[
    "oracle_gain"
].mean()

oracle_gap = (
    mean_oracle
    - mean_uncertainty
)

harm_random = (
    df1[
        "random_harm_probability"
    ].mean()
    * 100
)

harm_uncertainty = (
    df1[
        "official_harmful"
    ].mean()
    * 100
)

harm_oracle = (
    (df1["oracle_gain"] < 0).mean()
    * 100
)

official_gain = df1[
    "official_gain"
].to_numpy(dtype=float)

negative_mask = (
    official_gain < 0
)

zero_mask = (
    official_gain == 0
)

positive_mask = (
    official_gain > 0
)

negative_pct = (
    negative_mask.mean()
    * 100
)

zero_pct = (
    zero_mask.mean()
    * 100
)

positive_pct = (
    positive_mask.mean()
    * 100
)

full_min = float(
    official_gain.min()
)

full_max = float(
    official_gain.max()
)

# ============================================================
# Histogram display range
# ============================================================
nonzero_gain = official_gain[
    official_gain != 0
]

display_low = float(
    np.quantile(
        nonzero_gain,
        0.01
    )
)

display_high = float(
    np.quantile(
        nonzero_gain,
        0.99
    )
)

display_low = min(
    display_low,
    0
)

display_high = max(
    display_high,
    0
)

outside_mask = (
    (official_gain < display_low)
    |
    (official_gain > display_high)
)

outside_count = int(
    outside_mask.sum()
)

display_negative = official_gain[
    negative_mask
    & ~outside_mask
]

display_positive = official_gain[
    positive_mask
    & ~outside_mask
]

# ============================================================
# FIGURE 3.1
# ============================================================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.8, 4.15)
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    top=0.84,
    bottom=0.16,
    wspace=0.34
)

# ============================================================
# Figure 3.1(a)
# ============================================================
ax = axes[0]

style_axis(
    ax,
    "y"
)

panel_label(
    ax,
    "a"
)

labels = [
    "Random\n(expected)",
    "Uncertainty\nTop-1",
    "Oracle\n(offline)"
]

x = np.array([
    0.0,
    1.20,
    2.40
])

gain_values = [
    mean_random,
    mean_uncertainty,
    mean_oracle
]

gain_colors = [
    SOFT_BLUE,
    SOFT_PINK,
    MID_PINK
]

bars = ax.bar(
    x,
    gain_values,
    width=0.70,
    color=gain_colors,
    edgecolor="none"
)

ax.axhline(
    0,
    color="#777777",
    linewidth=0.9
)

ax.set_title(
    "Mean intervention gain",
    pad=12
)

ax.set_ylabel(
    "Reduction in normalized pose error"
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(
    -0.0032,
    0.0074
)

ax.set_xlim(
    -0.65,
    3.05
)

ax.yaxis.set_major_locator(
    MultipleLocator(0.002)
)

ax.yaxis.set_major_formatter(
    FormatStrFormatter("%.3f")
)

for bar, value in zip(
    bars,
    gain_values
):
    xpos = (
        bar.get_x()
        + bar.get_width() / 2
    )

    if value < 0:
        ypos = value - 0.00022
        va = "top"
    else:
        ypos = value + 0.00020
        va = "bottom"

    ax.text(
        xpos,
        ypos,
        f"{value:+.6f}",
        ha="center",
        va=va,
        fontsize=9,
        color=TEXT
    )

# ============================================================
# Cleaner gap annotation
# ============================================================
gap_x = 1.83

ax.annotate(
    "",
    xy=(
        gap_x,
        mean_oracle
    ),
    xytext=(
        gap_x,
        mean_uncertainty
    ),
    arrowprops=dict(
        arrowstyle="<->",
        color=MUTED,
        lw=1.05
    )
)

# Move text further LEFT
ax.text(
    1.72,
    (
        mean_oracle
        + mean_uncertainty
    ) / 2,
    (
        "Oracle − Top-1\n"
        f"= {oracle_gap:.6f}"
    ),
    ha="right",
    va="center",
    fontsize=8.2,
    color=MUTED,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.95,
        pad=1.0
    )
)

# ============================================================
# Figure 3.1(b)
# ============================================================
ax = axes[1]

style_axis(
    ax,
    "y"
)

panel_label(
    ax,
    "b"
)

harm_values = [
    harm_random,
    harm_uncertainty,
    harm_oracle
]

bars = ax.bar(
    x,
    harm_values,
    width=0.70,
    color=gain_colors,
    edgecolor="none"
)

ax.set_title(
    "Harmful intervention rate",
    pad=12
)

ax.set_ylabel(
    "Person instances (%)"
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlim(
    -0.65,
    3.05
)

ax.set_ylim(
    0,
    70
)

ax.yaxis.set_major_locator(
    MultipleLocator(10)
)

for bar, value in zip(
    bars,
    harm_values
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 1.5,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color=TEXT
    )

# ============================================================
# Figure 3.1(c)
# ============================================================
ax = axes[2]

style_axis(
    ax,
    "y"
)

panel_label(
    ax,
    "c"
)

negative_edges = np.linspace(
    display_low,
    0,
    22
)

positive_edges = np.linspace(
    0,
    display_high,
    22
)

negative_weights = np.full(
    len(display_negative),
    100.0 / N
)

positive_weights = np.full(
    len(display_positive),
    100.0 / N
)

ax.hist(
    display_negative,
    bins=negative_edges,
    weights=negative_weights,
    color=MID_BLUE,
    alpha=0.88,
    edgecolor="none",
    label="Negative gain"
)

ax.hist(
    display_positive,
    bins=positive_edges,
    weights=positive_weights,
    color=SOFT_PINK,
    alpha=0.95,
    edgecolor="none",
    label="Positive gain"
)

bin_width_reference = min(
    abs(display_low),
    abs(display_high)
)

zero_width = (
    bin_width_reference
    / 14
)

ax.bar(
    0,
    zero_pct,
    width=zero_width,
    color=ZERO_YELLOW,
    edgecolor="none",
    zorder=4,
    label="Zero gain"
)

ax.axvline(
    0,
    color="#777777",
    linestyle="--",
    linewidth=0.9,
    zorder=5
)

ax.set_title(
    "Gain distribution: Uncertainty Top-1",
    pad=12
)

ax.set_xlabel(
    "Reduction in normalized pose error"
)

ax.set_ylabel(
    "Proportion of all instances (%)"
)

ax.set_xlim(
    display_low,
    display_high
)

ax.text(
    0.025,
    0.96,
    (
        f"Negative: {negative_pct:.2f}%\n"
        f"Zero: {zero_pct:.2f}%\n"
        f"Positive: {positive_pct:.2f}%"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.6,
    color=TEXT,
    bbox=dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor="#DDDDDD",
        linewidth=0.6,
        alpha=0.94
    )
)

ax.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.00, 0.80),
    fontsize=8.3
)

# ============================================================
# Save Figure 3.1
# ============================================================
fig31_png = os.path.join(
    out_dir,
    "Figure_3_1_selection_final_v5.png"
)

fig31_pdf = os.path.join(
    out_dir,
    "Figure_3_1_selection_final_v5.pdf"
)

fig.savefig(
    fig31_png,
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    fig31_pdf,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# FIGURE 3.2 DATA
# ============================================================
g025 = df2[
    "gain_alpha_0_25"
].to_numpy(dtype=float)

g050 = df2[
    "gain_alpha_0_5"
].to_numpy(dtype=float)

g075 = df2[
    "gain_alpha_0_75"
].to_numpy(dtype=float)

g100 = df2[
    "gain_alpha_1_0"
].to_numpy(dtype=float)

partial_best = np.maximum.reduce([
    g025,
    g050,
    g075
])

partial_better = (
    partial_best
    > g100 + 1e-12
)

partial_rate = (
    partial_better.mean()
    * 100
)

full_rate = (
    100
    - partial_rate
)

alpha_values = np.array([
    0.25,
    0.50,
    0.75,
    1.00
])

alpha_gains = [
    g025,
    g050,
    g075,
    g100
]

mean_gains = np.array([
    values.mean()
    for values in alpha_gains
])

mean_gains_scaled = (
    mean_gains
    * 1e5
)

# ============================================================
# Full-data summary
# ============================================================
best_positive = np.maximum.reduce([
    g025,
    g050,
    g075,
    g100
])

oracle_strength_gap = (
    best_positive
    - g100
)

mean_gap = float(
    oracle_strength_gap.mean()
)

se = (
    oracle_strength_gap.std(ddof=1)
    / np.sqrt(
        len(
            oracle_strength_gap
        )
    )
)

ci_low = (
    mean_gap
    - 1.96 * se
)

ci_high = (
    mean_gap
    + 1.96 * se
)

# ============================================================
# Relative heatmap
# ============================================================
all_difference = np.column_stack([
    g025 - g100,
    g050 - g100,
    g075 - g100
])

rng = np.random.default_rng(
    42
)

sample_n = 120

sample_indices = rng.choice(
    len(df2),
    size=sample_n,
    replace=False
)

heat = all_difference[
    sample_indices
].copy()

best_partial_index = np.argmax(
    heat,
    axis=1
)

best_partial_difference = np.max(
    heat,
    axis=1
)

order = np.lexsort((
    -best_partial_difference,
    best_partial_index
))

heat = heat[
    order
]

best_partial_index = best_partial_index[
    order
]

# 98th percentile saturation
abs_all_difference = np.abs(
    all_difference.ravel()
)

colour_limit = float(
    np.quantile(
        abs_all_difference,
        0.98
    )
)

heat_norm = Normalize(
    vmin=-colour_limit,
    vmax=colour_limit,
    clip=True
)

# ============================================================
# FIGURE 3.2
# Give panel (a) slightly MORE width
# ============================================================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.8, 4.20),
    gridspec_kw={
        "width_ratios": [
            1.18,
            1.00,
            1.62
        ]
    }
)

fig.subplots_adjust(
    left=0.06,
    right=0.97,
    top=0.84,
    bottom=0.16,
    wspace=0.38
)

# ============================================================
# Figure 3.2(a)
# ============================================================
ax = axes[0]

style_axis(
    ax,
    "x"
)

panel_label(
    ax,
    "a"
)

# Slightly thicker bar
BAR_HEIGHT = 0.28

ax.barh(
    [0],
    [full_rate],
    height=BAR_HEIGHT,
    color=MID_PINK,
    edgecolor="none"
)

ax.barh(
    [0],
    [partial_rate],
    left=[full_rate],
    height=BAR_HEIGHT,
    color=SOFT_BLUE,
    edgecolor="none"
)

ax.set_xlim(
    0,
    100
)

ax.set_ylim(
    -0.34,
    0.34
)

ax.set_yticks([])

ax.set_xlabel(
    "Person instances (%)"
)

ax.set_title(
    "Is full replacement optimal?",
    pad=12
)

ax.xaxis.set_major_locator(
    MultipleLocator(20)
)

# ------------------------------------------------------------
# FINAL text layout:
# 3 lines on BOTH sides.
# ------------------------------------------------------------
ax.text(
    full_rate / 2,
    0,
    (
        "Full replacement\n"
        "best / tied\n"
        f"{full_rate:.2f}%"
    ),
    ha="center",
    va="center",
    fontsize=8.7,
    color="white",
    fontweight="bold",
    linespacing=1.10
)

ax.text(
    full_rate
    + partial_rate / 2,
    0,
    (
        "Partial\n"
        "strictly better\n"
        f"{partial_rate:.2f}%"
    ),
    ha="center",
    va="center",
    fontsize=8.7,
    color=TEXT,
    fontweight="bold",
    linespacing=1.10
)

# ============================================================
# Figure 3.2(b)
# ============================================================
ax = axes[1]

style_axis(
    ax,
    "y"
)

panel_label(
    ax,
    "b"
)

x = np.arange(
    len(alpha_values)
)

bars = ax.bar(
    x,
    mean_gains_scaled,
    width=0.58,
    color=SOFT_PINK,
    edgecolor="none"
)

ax.axhline(
    0,
    color="#777777",
    linewidth=0.9
)

ax.set_xticks(x)

ax.set_xticklabels([
    "0.25",
    "0.50",
    "0.75",
    "1.00"
])

ax.set_xlabel(
    r"Intervention strength $\alpha$"
)

ax.set_ylabel(
    r"Mean gain ($\times 10^{-5}$)"
)

ax.set_title(
    "Mean gain of fixed strengths",
    pad=12
)

lowest_scaled = float(
    mean_gains_scaled.min()
)

ax.set_ylim(
    lowest_scaled - 1.7,
    1.2
)

for bar, value in zip(
    bars,
    mean_gains_scaled
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value - 0.28,
        f"{value:.2f}",
        ha="center",
        va="top",
        fontsize=9.2,
        color=TEXT
    )

# ============================================================
# Figure 3.2(c)
# ============================================================
ax = axes[2]

panel_label(
    ax,
    "c"
)

im = ax.imshow(
    heat,
    aspect="auto",
    cmap=diff_cmap,
    norm=heat_norm,
    interpolation="nearest"
)

ax.set_title(
    "Gain difference relative to full replacement",
    pad=12
)

ax.set_xticks([
    0,
    1,
    2
])

ax.set_xticklabels([
    r"$\alpha=0.25$",
    r"$\alpha=0.50$",
    r"$\alpha=0.75$"
])

ax.set_xlabel(
    "Partial intervention strength"
)

ax.set_ylabel(
    "Sampled person instances"
)

ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(
    length=0,
    labelcolor=TEXT
)

# Group boundaries
group_change = np.where(
    np.diff(
        best_partial_index
    ) != 0
)[0]

for boundary in group_change:
    ax.axhline(
        boundary + 0.5,
        color="white",
        linewidth=1.2,
        alpha=0.95
    )

# Colour bar
cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.028,
    extend="both"
)

cbar.set_label(
    "Gain difference relative to full replacement"
)

cbar.outline.set_visible(False)

# ============================================================
# Save Figure 3.2
# ============================================================
fig32_png = os.path.join(
    out_dir,
    "Figure_3_2_strength_final_v5.png"
)

fig32_pdf = os.path.join(
    out_dir,
    "Figure_3_2_strength_final_v5.pdf"
)

fig.savefig(
    fig32_png,
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    fig32_pdf,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# Check
# ============================================================
print()
print("=" * 72)
print("V5 FINAL FIGURES GENERATED")
print("=" * 72)

print()
print("Figure 3.1")
print(
    f"Oracle - Top-1 gap      = "
    f"{oracle_gap:+.8f}"
)
print(
    f"Neg / Zero / Pos        = "
    f"{negative_pct:.2f}% / "
    f"{zero_pct:.2f}% / "
    f"{positive_pct:.2f}%"
)
print(
    f"Histogram range         = "
    f"[{display_low:+.8f}, "
    f"{display_high:+.8f}]"
)
print(
    f"Outside range           = "
    f"{outside_count}"
)
print(
    f"Full range              = "
    f"[{full_min:+.8f}, "
    f"{full_max:+.8f}]"
)

print()
print("Figure 3.2")
print(
    f"Full best / tied        = "
    f"{full_rate:.2f}%"
)
print(
    f"Partial strictly better = "
    f"{partial_rate:.2f}%"
)
print(
    f"Mean BestPositive-a1    = "
    f"{mean_gap:+.8f}"
)
print(
    f"95% CI                  = "
    f"[{ci_low:+.8f}, "
    f"{ci_high:+.8f}]"
)
print(
    f"Heatmap saturation      = "
    f"±{colour_limit:.8f}"
)

print()
print("Saved:")
print(fig31_png)
print(fig31_pdf)
print(fig32_png)
print(fig32_pdf)
