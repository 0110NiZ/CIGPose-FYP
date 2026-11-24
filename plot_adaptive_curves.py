import re
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


WORK_DIR = Path("work_dirs/group_e_adaptive_50pct_t01")
OUT_DIR = Path("adaptive_curve_plots")
OUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 1. 自动寻找训练日志
# =========================================================
logs = list(WORK_DIR.rglob("*.log"))

if not logs:
    raise FileNotFoundError(
        f"No .log files found under {WORK_DIR}"
    )

log_path = max(logs, key=lambda p: p.stat().st_mtime)

print("Using log:")
print(log_path)


# =========================================================
# 2. 读取 LOSS_CURVE 和 validation 指标
# =========================================================
loss_records = []
val_records = []

loss_pattern = re.compile(r'LOSS_CURVE\s+(\{.*\})')

val_pattern = re.compile(
    r'Epoch\(val\)\s+\[(\d+)\]\[\d+/\d+\].*?'
    r'coco/AP:\s*([0-9.eE+-]+).*?'
    r'coco/AP \.5:\s*([0-9.eE+-]+).*?'
    r'coco/AP \.75:\s*([0-9.eE+-]+).*?'
    r'occluded/mean_px:\s*([0-9.eE+-]+).*?'
    r'visible/mean_px:\s*([0-9.eE+-]+)'
)

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:

        # LOSS_CURVE
        m = loss_pattern.search(line)

        if m:
            try:
                record = json.loads(m.group(1))
                loss_records.append(record)
            except Exception:
                pass

        # Validation metrics
        m = val_pattern.search(line)

        if m:
            val_records.append({
                "epoch": int(m.group(1)),
                "AP": float(m.group(2)) * 100,
                "AP50": float(m.group(3)) * 100,
                "AP75": float(m.group(4)) * 100,
                "occluded_px": float(m.group(5)),
                "visible_px": float(m.group(6)),
            })


loss_df = pd.DataFrame(loss_records)
val_df = pd.DataFrame(val_records)

if not loss_df.empty:
    loss_df = (
        loss_df
        .drop_duplicates("epoch", keep="last")
        .sort_values("epoch")
    )

if not val_df.empty:
    val_df = (
        val_df
        .drop_duplicates("epoch", keep="last")
        .sort_values("epoch")
    )


print("\nFound LOSS_CURVE epochs:", len(loss_df))
print("Found validation epochs:", len(val_df))


# =========================================================
# 3. 保存 CSV
# =========================================================
loss_df.to_csv(
    OUT_DIR / "training_curve.csv",
    index=False
)

val_df.to_csv(
    OUT_DIR / "validation_curve.csv",
    index=False
)


def anneal_line():
    # 你们老师设置：最后20轮开始退火
    plt.axvline(
        80,
        linestyle="--",
        linewidth=1.5,
        label="KD annealing starts (epoch 80)"
    )


# =========================================================
# 4. Train / Validation keypoint loss
# =========================================================
if not loss_df.empty:

    plt.figure(figsize=(10, 6))

    if "train_loss_kpt_per_person" in loss_df.columns:
        plt.plot(
            loss_df["epoch"],
            loss_df["train_loss_kpt_per_person"],
            label="Train keypoint loss"
        )

    if "val_loss_kpt_per_person" in loss_df.columns:
        plt.plot(
            loss_df["epoch"],
            loss_df["val_loss_kpt_per_person"],
            label="Validation keypoint loss"
        )

    anneal_line()

    plt.xlabel("Epoch")
    plt.ylabel("Loss per person")
    plt.title("Keypoint Loss Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "01_keypoint_loss.png",
        dpi=200
    )

    plt.close()


# =========================================================
# 5. SimCC KD loss
# =========================================================
if (
    not loss_df.empty
    and "train_weighted_kd_batch_mean" in loss_df.columns
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        loss_df["epoch"],
        loss_df["train_weighted_kd_batch_mean"],
        label="Weighted SimCC KD loss"
    )

    anneal_line()

    plt.xlabel("Epoch")
    plt.ylabel("KD loss")
    plt.title("SimCC Knowledge Distillation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "02_simcc_kd_loss.png",
        dpi=200
    )

    plt.close()


# =========================================================
# 6. AP / AP50 / AP75
# =========================================================
if not val_df.empty:

    plt.figure(figsize=(10, 6))

    plt.plot(
        val_df["epoch"],
        val_df["AP"],
        marker="o",
        label="AP"
    )

    plt.plot(
        val_df["epoch"],
        val_df["AP50"],
        marker="o",
        label="AP50"
    )

    plt.plot(
        val_df["epoch"],
        val_df["AP75"],
        marker="o",
        label="AP75"
    )

    anneal_line()

    plt.xlabel("Epoch")
    plt.ylabel("AP (%)")
    plt.title("COCO Validation AP")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "03_validation_AP.png",
        dpi=200
    )

    plt.close()


# =========================================================
# 7. 遮挡 / 可见关节误差
# =========================================================
if not val_df.empty:

    plt.figure(figsize=(10, 6))

    plt.plot(
        val_df["epoch"],
        val_df["occluded_px"],
        marker="o",
        label="Occluded joints"
    )

    plt.plot(
        val_df["epoch"],
        val_df["visible_px"],
        marker="o",
        label="Visible joints"
    )

    anneal_line()

    plt.xlabel("Epoch")
    plt.ylabel("Mean pixel error (px)")
    plt.title("Joint Localization Error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "04_joint_error.png",
        dpi=200
    )

    plt.close()


# =========================================================
# 8. 单独画遮挡误差
#    因为 visible 大约6px，和 occluded 约16px 放一起
#    不太容易观察遮挡后期的小变化
# =========================================================
if not val_df.empty:

    plt.figure(figsize=(10, 6))

    plt.plot(
        val_df["epoch"],
        val_df["occluded_px"],
        marker="o",
        label="Occluded joint error"
    )

    anneal_line()

    plt.xlabel("Epoch")
    plt.ylabel("Mean pixel error (px)")
    plt.title("Occluded Joint Error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "05_occluded_error.png",
        dpi=200
    )

    plt.close()


# =========================================================
# 9. 打印最后结果
# =========================================================
print("\n==============================")
print("VALIDATION CURVE")
print("==============================")

if not val_df.empty:
    print(
        val_df[
            [
                "epoch",
                "AP",
                "AP50",
                "AP75",
                "occluded_px",
                "visible_px"
            ]
        ].to_string(index=False)
    )

    last = val_df.iloc[-1]

    print("\nFinal:")
    print(f"AP              = {last['AP']:.4f}")
    print(f"AP50            = {last['AP50']:.4f}")
    print(f"AP75            = {last['AP75']:.4f}")
    print(
        f"Occluded error  = "
        f"{last['occluded_px']:.4f} px"
    )
    print(
        f"Visible error   = "
        f"{last['visible_px']:.4f} px"
    )


print("\nPlots saved to:")
print(OUT_DIR.resolve())
