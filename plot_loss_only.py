import re
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


WORK_DIR = Path("work_dirs/group_e_adaptive_50pct_t01")
OUT_DIR = Path("adaptive_curve_plots")
OUT_DIR.mkdir(exist_ok=True)

# 自动找最新 log
logs = list(WORK_DIR.rglob("*.log"))
if not logs:
    raise FileNotFoundError(f"No .log files found under {WORK_DIR}")

log_path = max(logs, key=lambda p: p.stat().st_mtime)

print("Using log:")
print(log_path)

loss_records = []
pattern = re.compile(r'LOSS_CURVE\s+(\{.*\})')

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            try:
                record = json.loads(m.group(1))
                loss_records.append(record)
            except Exception:
                pass

if not loss_records:
    raise RuntimeError("No LOSS_CURVE records found in the log.")

df = pd.DataFrame(loss_records)
df = df.drop_duplicates("epoch", keep="last").sort_values("epoch")

csv_path = OUT_DIR / "loss_curve_only.csv"
df.to_csv(csv_path, index=False)

print("\nFound epochs:")
print(df[["epoch", "train_loss_kpt_per_person", "val_loss_kpt_per_person"]].to_string(index=False))

# 画图
plt.figure(figsize=(10, 6))
plt.plot(
    df["epoch"],
    df["train_loss_kpt_per_person"],
    marker="o",
    label="Train keypoint loss / person"
)
plt.plot(
    df["epoch"],
    df["val_loss_kpt_per_person"],
    marker="o",
    label="Validation keypoint loss / person"
)

# 退火起点
plt.axvline(
    80,
    linestyle="--",
    linewidth=1.5,
    label="KD annealing starts (epoch 80)"
)

plt.xlabel("Epoch")
plt.ylabel("Loss per person")
plt.title("Train vs Validation Keypoint Loss")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

png_path = OUT_DIR / "06_train_val_loss.png"
plt.savefig(png_path, dpi=200)
plt.close()

print("\nSaved:")
print(csv_path.resolve())
print(png_path.resolve())
