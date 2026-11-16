#!/usr/bin/env python3
"""Read-only, reproducible summary of Module A signed Gain NPZ labels.

This script uses saved label shards only: it never loads CIGPose, changes
training labels, or re-infers candidates. It is an analysis, not an audit
substitute. Example:

    python mmpose/analysis/fyp/summarize_a_labels.py \
        --output /path/to/train_a_labels --expect-full

Dependencies: numpy; matplotlib for plots (omit with --no-plots).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


JOINTS = (
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
)
REQUIRED = frozenset({"image_id", "annotation_id", "oks_before", "oks_after", "gain"})


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, records):
    records = list(records)
    if not records:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def generate_plots(destination: Path, gain, joint_records, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gain.ravel(), bins=120)
    ax.set_title("Train2017: Distribution of All True Gains")
    ax.set_xlabel("True OKS Gain")
    ax.set_ylabel("Number of Joint Candidates (log scale)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(destination / "gain_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(JOINTS, [r["mean_gain"] for r in joint_records])
    ax.axhline(0, linewidth=1)
    ax.set_title("Mean True Gain by COCO Joint")
    ax.set_ylabel("Mean OKS Gain")
    ax.tick_params(axis="x", labelrotation=65)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(destination / "mean_gain_per_joint.png", dpi=180)
    plt.close(fig)

    freq = np.bincount(chosen, minlength=17) * 100.0 / len(chosen)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(JOINTS, freq)
    ax.set_title("Oracle Best Joint Frequency (offline label analysis)")
    ax.set_ylabel("Percentage of Persons (%)")
    ax.tick_params(axis="x", labelrotation=65)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(destination / "oracle_joint_frequency.png", dpi=180)
    plt.close(fig)


def summarize(output: Path, destination: Path, expect_full: bool, no_plots: bool):
    status = read_json(output / "status.json")
    report = read_json(output / "validation_report.json")
    done = status.get("done", {})
    if status.get("failed", {}):
        raise ValueError("Failed source shards remain; fix/retry before summarizing")
    if not report.get("passed", False):
        raise ValueError("Independent validation_report.json is not passed")

    files = sorted(output.glob("shard-*.npz"))
    declared = {record["file"]: record for record in done.values()}
    if len(declared) != len(done) or {p.name for p in files} != set(declared):
        raise ValueError("Output NPZ files and status.done manifest do not agree")

    image_ids, ann_ids, before_parts, after_parts, gain_parts = [], [], [], [], []
    for index, path in enumerate(files, start=1):
        rec = declared[path.name]
        if sha256_file(path) != rec["sha256"]:
            raise ValueError(f"File SHA256 mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != REQUIRED:
                raise ValueError(f"Wrong NPZ fields: {path.name}")
            im = z["image_id"]
            ann = z["annotation_id"]
            before = z["oks_before"]
            after = z["oks_after"]
            gain = z["gain"]
        n = len(ann)
        if not (
            im.shape == (n,) and ann.shape == (n,) and before.shape == (n,)
            and after.shape == (n, 17) and gain.shape == (n, 17)
            and im.dtype == np.dtype("int64") and ann.dtype == np.dtype("int64")
            and all(np.issubdtype(a.dtype, np.floating) for a in (before, after, gain))
            and n == int(rec["count"])
        ):
            raise ValueError(f"Label shape/count/type mismatch: {path.name}")
        if not (np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(gain).all()):
            raise ValueError(f"Non-finite labels: {path.name}")
        if not (np.all((before >= 0) & (before <= 1)) and np.all((after >= 0) & (after <= 1))):
            raise ValueError(f"OKS out of range: {path.name}")
        if not np.allclose(gain, after - before[:, None], atol=2e-6, rtol=0):
            raise ValueError(f"Gain arithmetic failed: {path.name}")
        image_ids.append(im)
        ann_ids.append(ann)
        before_parts.append(before.astype(np.float64))
        after_parts.append(after.astype(np.float64))
        gain_parts.append(gain.astype(np.float64))
        if index % 1000 == 0 or index == len(files):
            print(f"Checked {index}/{len(files)} label shards", flush=True)

    images = np.concatenate(image_ids)
    anns = np.concatenate(ann_ids)
    before = np.concatenate(before_parts)
    after = np.concatenate(after_parts)
    gain = np.concatenate(gain_parts)
    n = len(anns)
    if len(np.unique(anns)) != n:
        raise ValueError("Duplicate annotation_id in labels")
    if n != int(report.get("verified_persons", -1)) or len(files) != int(report.get("verified_shards", -1)):
        raise ValueError("Report count does not agree with labels")
    if expect_full and not (n == 149813 and len(files) == 4682 and report.get("coverage_complete") is True):
        raise ValueError("Expected full Train2017 coverage not confirmed")

    best_joint = np.argmax(gain, axis=1)
    best_gain = gain[np.arange(n), best_joint]
    best_after = after[np.arange(n), best_joint]
    total = gain.size
    positives = int(np.count_nonzero(gain > 0))
    negatives = int(np.count_nonzero(gain < 0))
    zeros = int(np.count_nonzero(gain == 0))
    positive_persons = int(np.count_nonzero(best_gain > 0))
    zero_persons = int(np.count_nonzero(best_gain == 0))
    negative_persons = int(np.count_nonzero(best_gain < 0))

    summary = {
        "dataset": "COCO train2017 GT-box",
        "baseline": "NIE cached P_0_coords",
        "persons": n, "shards": len(files), "gain_values": total,
        "positive": positives, "negative": negatives, "zero": zeros,
        "positive_pct": positives / total * 100,
        "negative_pct": negatives / total * 100,
        "zero_pct": zeros / total * 100,
        "persons_with_positive_candidate": positive_persons,
        "persons_with_zero_best_gain": zero_persons,
        "persons_with_negative_best_gain": negative_persons,
        "persons_with_positive_candidate_pct": positive_persons / n * 100,
        "baseline_oks": describe(before),
        "all_candidate_oks": describe(after),
        "gain": describe(gain),
        "oracle_best_gain": describe(best_gain),
        "oracle_best_candidate_oks": describe(best_after),
    }
    joint_records = []
    best_counts = np.bincount(best_joint, minlength=17)
    for j, label in enumerate(JOINTS):
        col = gain[:, j]
        joint_records.append({
            "joint_id": j,
            "joint_name": label,
            "mean_gain": float(col.mean()),
            "median_gain": float(np.median(col)),
            "std_gain": float(col.std()),
            "positive_pct": float(np.mean(col > 0) * 100),
            "negative_pct": float(np.mean(col < 0) * 100),
            "zero_pct": float(np.mean(col == 0) * 100),
            "oracle_best_count": int(best_counts[j]),
            "oracle_best_pct": float(best_counts[j] / n * 100),
        })

    case_records = []
    selected = [("highest_oracle_gain", i) for i in np.argsort(best_gain)[-10:][::-1]]
    selected += [("lowest_oracle_gain", i) for i in np.argsort(best_gain)[:10]]
    for kind, index in selected:
        case_records.append({
            "case_type": kind,
            "image_id": int(images[index]),
            "annotation_id": int(anns[index]),
            "oks_before": float(before[index]),
            "oracle_best_joint": int(best_joint[index]),
            "oracle_best_gain": float(best_gain[index]),
            "oracle_best_oks_after": float(best_after[index]),
        })

    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "gain_summary.json", summary)
    write_csv(destination / "per_joint_gain.csv", joint_records)
    write_csv(destination / "representative_case_ids.csv", case_records)
    report_md = [
        "# Module A | Train2017 True Gain Summary", "",
        "## Provenance", "",
        "- Labels: Train2017 GT-box, 17 independent alpha=1 candidates per person.",
        "- Official P0: NIE shared cache `P_0_coords`.",
        "- Rebuilt P0: from unchanged NIE cache `features` using TANG's common prediction/decoding interface; diagnostic only.",
        "- This is **not** a NIE P0 vs independently stored TANG P0 comparison.",
        "", "## Coverage and audit", "",
        f"- Shards: {len(files):,}; people: {n:,}; Gains: {total:,}",
        f"- Independent audit passed: {report['passed']}; coverage complete: {report.get('coverage_complete')}",
        "- Post-hoc summary checks NPZ fields, hashes, counts, signs and Gain arithmetic; it does not rerun candidate inference.",
        "", "## Gain signs", "",
        f"- Positive: {positives:,} ({summary['positive_pct']:.2f}%)",
        f"- Negative: {negatives:,} ({summary['negative_pct']:.2f}%)",
        f"- Zero: {zeros:,} ({summary['zero_pct']:.2f}%)",
        f"- Mean true Gain: {summary['gain']['mean']:+.8f}",
        "", "## Per-person oracle label analysis (NOT learned performance)", "",
        f"- People with positive candidate: {positive_persons:,}/{n:,} ({positive_persons/n*100:.2f}%)",
        f"- Mean oracle-best gain: {summary['oracle_best_gain']['mean']:+.8f}",
        f"- Mean NIE cached P0 OKS: {summary['baseline_oks']['mean']:.8f}",
        f"- Mean oracle-best candidate OKS: {summary['oracle_best_candidate_oks']['mean']:.8f}",
        "", "## Limitations", "",
        "- Some NIE cached P0 vs rebuilt P0 differences remain unexplained; see review_a_labels results.",
        "- No Train2017 true Gain or GT may enter predictor inference inputs.",
        "- No claim is made about COCO val AP/AR or A predictor training.", "",
    ]
    (destination / "GAIN_RESULTS_SUMMARY.md").write_text("\n".join(report_md), encoding="utf-8")
    if not no_plots:
        generate_plots(destination, gain, joint_records, best_joint)
    print(f"PASS: {n:,} people, {total:,} Gain labels, {len(files)} shards")
    print(f"Gain signs: +{positives:,} / -{negatives:,} / zero {zeros:,}")
    print(f"Mean gain: {summary['gain']['mean']:+.8f}")
    print(f"Wrote: {destination}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Full/partial label output directory")
    parser.add_argument("--save-dir", type=Path, help="Default: OUTPUT/gain_statistics")
    parser.add_argument("--expect-full", action="store_true", help="Require 4682 shards, 149813 persons and full coverage")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    dest = args.save_dir.expanduser().resolve() if args.save_dir else output / "gain_statistics"
    summarize(output, dest, args.expect_full, args.no_plots)


if __name__ == "__main__":
    main()
