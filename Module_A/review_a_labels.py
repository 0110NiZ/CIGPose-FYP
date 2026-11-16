#!/usr/bin/env python3
"""Module A | independent label-review and visual evidence.

Read-only relative to the original label shards. Works with Y's validated
build_a_labels.py and N's GT-box train2017 cache.

Outputs under <output>/review_a_labels/:
  p0_difference_summary.json, p0_top_shards.csv,
  p0_outlier_cases.csv, p0_outliers/*.png,
  gain_cases.csv, gain_cases/*.png,
  semantic_checks.json, REVIEW_REPORT.md.

Usage:
  python review_a_labels.py --repo REPO --cache TRAIN2017_CACHE \
      --checkpoint CIGPOSE_CHECKPOINT --output TRAIN_A_LABELS \
      --device cuda:0 [--images-dir COCO_TRAIN2017_IMAGES]

IMPORTANT: NIE cached P_0_coords is the official Gain baseline. Rebuilt P0 is
obtained by applying TANG's shared predict/decode path to NIE cached FEATURES.
This is NOT a comparison of two independently saved NIE/TANG P0 datasets.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]):
    if not records:
        return
    columns = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def aggregate_p0(output: Path, status: dict, report: dict, dest: Path):
    done = status.get("done", {})
    failed = status.get("failed", {})
    if len(done) != 4682 or failed:
        raise RuntimeError(f"Expected 4682 Done / 0 Failed; got {len(done)} / {len(failed)}")
    if report.get("passed") is not True or report.get("coverage_complete") is not True:
        raise RuntimeError("Full independent audit must pass before review")
    if int(report.get("verified_persons", -1)) != 149813:
        raise RuntimeError("Full audit verified_persons mismatch")

    records = []
    for source_name, entry in sorted(done.items()):
        path = output / "p0_diagnostics" / (Path(source_name).stem + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"P0 diagnostic missing: {path}")
        d = read_json(path)
        n = int(entry["count"])
        if d["source_shard"] != source_name or int(d["persons"]) != n:
            raise RuntimeError(f"P0 diagnostic identity/count mismatch: {source_name}")
        records.append({
            "source_shard": source_name,
            "label_file": entry["file"],
            "persons": n,
            "max_coord_px": float(d["p0_max_abs_error_px"]),
            "mean_coord_px": float(d["p0_mean_abs_error_px"]),
            "p95_coord_px_within_shard": float(d["p0_p95_abs_error_px"]),
            "max_abs_oks_difference": float(d["oks_max_abs_difference"]),
            "mean_abs_oks_difference": float(d["oks_mean_abs_difference"]),
            "worst_person_index": int(d["worst_person_index"]),
            "worst_joint": int(d["worst_joint"]),
            "worst_axis": int(d["worst_axis"]),
        })

    n_people = sum(r["persons"] for r in records)
    if n_people != 149813:
        raise RuntimeError(f"Total persons mismatch: {n_people}")
    worst = sorted(records, key=lambda r: r["max_coord_px"], reverse=True)
    oks_worst = sorted(records, key=lambda r: r["max_abs_oks_difference"], reverse=True)
    summary = {
        "official_baseline": "NIE cached P_0_coords",
        "rebuilt_p0_definition": "Decode unmodified NIE cached features via TANG's public prediction/decoding interface",
        "comparison_is_not": "NIE P0 versus independently saved TANG P0",
        "source_shards": len(records),
        "persons": n_people,
        "coordinate_components_per_person": 34,
        "mean_absolute_coordinate_difference_px": sum(r["mean_coord_px"] * r["persons"] for r in records) / n_people,
        "mean_absolute_instance_oks_difference": sum(r["mean_abs_oks_difference"] * r["persons"] for r in records) / n_people,
        "maximum_absolute_coordinate_difference_px": worst[0]["max_coord_px"],
        "maximum_absolute_instance_oks_difference": oks_worst[0]["max_abs_oks_difference"],
        "shards_with_max_coordinate_difference_above_px": {
            str(t): sum(r["max_coord_px"] > t for r in records)
            for t in (0.1, 1.0, 5.0, 10.0, 20.0)
        },
        "important_limitations": [
            "Threshold counts are SHARD counts, not person/joint counts.",
            "Global means are weighted by the exact number of persons in each shard.",
            "Global coordinate median/percentiles cannot be inferred from per-shard quantiles.",
            "Original NIE cached SimCC logits are not available, so numerical tie causation is not conclusively proven.",
        ],
    }
    write_json(dest / "p0_difference_summary.json", summary)
    write_csv(dest / "p0_top_shards.csv", worst[:50])
    write_csv(dest / "p0_top_oks_shards.csv", oks_worst[:50])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coord_max = np.asarray([r["max_coord_px"] for r in records], dtype=np.float64)
    oks_max = np.asarray([r["max_abs_oks_difference"] for r in records], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log10(1 + coord_max), bins=65)
    ax.set_xlabel("log10(1 + shard maximum |P0 rebuilt - P0 cached| in pixels)")
    ax.set_ylabel("Number of source shards")
    ax.set_title("Train2017: Shard-level P0 Reconstruction Differences")
    fig.tight_layout()
    fig.savefig(dest / "p0_shard_max_hist.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(coord_max, oks_max, s=9, alpha=0.65)
    ax.set_xlabel("Shard max absolute coordinate difference (px)")
    ax.set_ylabel("Shard max absolute instance OKS difference")
    ax.set_title("P0 Coordinate Differences vs Instance OKS Differences")
    fig.tight_layout()
    fig.savefig(dest / "p0_shard_max_vs_oks.png", dpi=160)
    plt.close(fig)
    return summary, worst, oks_worst


def load_labels_for_cases(output: Path, status: dict):
    # The labels are ~2.5M floats, comfortably small enough for analysis.
    lookup = {r["file"]: s for s, r in status["done"].items()}
    file_paths = sorted(output.glob("shard-*.npz"))
    if len(file_paths) != len(lookup) or set(p.name for p in file_paths) != set(lookup):
        raise RuntimeError("Label files/status.done do not match")
    blocks = []
    boundaries = []
    gains = []
    before = []
    after = []
    ids = []
    image_ids = []
    offset = 0
    for p in file_paths:
        with np.load(p, allow_pickle=False) as z:
            arr = np.asarray(z["gain"], dtype=np.float64)
            n = len(arr)
            if arr.shape != (n, 17):
                raise ValueError(f"Invalid Gain shape in {p.name}")
            boundaries.append(offset)
            blocks.append((lookup[p.name], p.name, n))
            offset += n
            gains.append(arr)
            before.append(np.asarray(z["oks_before"], dtype=np.float64))
            after.append(np.asarray(z["oks_after"], dtype=np.float64))
            ids.append(np.asarray(z["annotation_id"], dtype=np.int64))
            image_ids.append(np.asarray(z["image_id"], dtype=np.int64))
    full = {
        "gain": np.concatenate(gains),
        "before": np.concatenate(before),
        "after": np.concatenate(after),
        "ids": np.concatenate(ids),
        "images": np.concatenate(image_ids),
    }
    if len(full["ids"]) != 149813 or len(np.unique(full["ids"])) != 149813:
        raise RuntimeError("Unexpected person count / duplicate IDs")
    if not np.isfinite(full["gain"]).all():
        raise RuntimeError("Nonfinite Gain in full labels")
    return full, blocks, boundaries


def summarize_true_gains(full: dict, dest: Path):
    gain = full["gain"]
    before = full["before"]
    after = full["after"]
    best_joint = np.argmax(gain, axis=1)
    best_gain = gain[np.arange(len(gain)), best_joint]
    counts = {
        "positive": int(np.count_nonzero(gain > 0)),
        "negative": int(np.count_nonzero(gain < 0)),
        "zero": int(np.count_nonzero(gain == 0)),
    }
    summary = {
        "persons": int(len(gain)),
        "candidates_per_person": 17,
        "total_candidate_gains": int(gain.size),
        "candidate_sign_counts": counts,
        "candidate_sign_pct": {key: value * 100 / gain.size for key, value in counts.items()},
        "mean_gain": float(gain.mean()),
        "median_gain": float(np.median(gain)),
        "p01_gain": float(np.percentile(gain, 1)),
        "p99_gain": float(np.percentile(gain, 99)),
        "mean_nie_cached_baseline_oks": float(before.mean()),
        "mean_oracle_best_gain": float(best_gain.mean()),
        "persons_with_at_least_one_positive_candidate": int(np.count_nonzero(best_gain > 0)),
        "oracle_best_is_analysis_only": True,
    }
    write_json(dest / "true_gain_summary.json", summary)
    rows = []
    for j in range(17):
        a = gain[:,j]
        rows.append({
            "joint_id": j,
            "mean_gain": float(a.mean()),
            "positive_pct": float(np.mean(a > 0) * 100),
            "negative_pct": float(np.mean(a < 0) * 100),
            "oracle_argmax_count": int(np.count_nonzero(best_joint == j)),
        })
    write_csv(dest / "per_joint_gain.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gain.ravel(), bins=120)
    ax.set_yscale("log")
    ax.set_xlabel("True OKS Gain, 17 candidate joints per person")
    ax.set_ylabel("Candidate count (log scale)")
    ax.set_title("Train2017: True Gain Distribution")
    fig.tight_layout()
    fig.savefig(dest / "true_gain_distribution.png", dpi=160)
    plt.close(fig)
    return summary


def choose_examples(full: dict, blocks, boundaries, count: int):
    gain = full["gain"]
    best_idx = np.argmax(gain, axis=1)
    worst_idx = np.argmin(gain, axis=1)
    best = gain[np.arange(len(gain)), best_idx]
    worst = gain[np.arange(len(gain)), worst_idx]
    # Percentile-like rank sampling among real signed candidates: medium,
    # strong and very strong. This avoids selecting only pathological extremes.
    used = set()
    items = []
    for sign, available, values, joints, fractions in [
        ("positive", np.flatnonzero(best > 0), best, best_idx, [0.50, 0.90, 0.99]),
        ("negative", np.flatnonzero(worst < 0), worst, worst_idx, [0.50, 0.10, 0.01]),
    ]:
        ordered = available[np.argsort(values[available])]
        if not len(ordered):
            continue
        if count <= 3:
            choices = fractions[:count]
        else:
            choices = np.linspace(0.2, 0.98, count).tolist() if sign == "positive" else np.linspace(0.8, 0.02, count).tolist()
        for fraction in choices:
            pivot = int(round(fraction * (len(ordered) - 1)))
            k = next((int(ordered[x]) for distance in range(len(ordered)) for x in (pivot-distance, pivot+distance) if 0 <= x < len(ordered) and int(ordered[x]) not in used), None)
            if k is None:
                continue
            used.add(k)
            shard_idx = bisect_right(boundaries, k) - 1
            src, label_file, _ = blocks[shard_idx]
            r = {
                "case_type": sign,
                "target_rank_fraction": float(fraction),
                "source_shard": src,
                "label_file": label_file,
                "person_index_in_shard": int(k - boundaries[shard_idx]),
                "image_id": int(full["images"][k]),
                "annotation_id": int(full["ids"][k]),
                "joint_id": int(joints[k]),
                "oks_before_saved": float(full["before"][k]),
                "oks_after_saved": float(full["after"][k, joints[k]]),
                "gain_saved": float(values[k]),
            }
            items.append(r)
    return items


# COCO keypoint edges (joint indices 0..16).
SKELETON = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,6),
            (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16)]


def draw_pose(ax, xy, color, name, visible=None, alpha=0.9):
    if visible is None:
        visible = np.ones(17, dtype=bool)
    xy = np.asarray(xy, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=color, linewidth=1.6, alpha=alpha)
    ax.scatter(xy[visible,0], xy[visible,1], s=15, color=color, label=name, alpha=alpha, zorder=3)


def draw_panels(destination: Path, title: str, panels: list[tuple], gt, original_size, image_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.asarray(gt, dtype=np.float64)
    gt_vis = gt[:,2] > 0
    coords = [gt[gt_vis,:2]] if gt_vis.any() else []
    coords.extend([np.asarray(p[1], dtype=np.float64) for p in panels])
    points = np.concatenate(coords)
    x0, y0 = np.min(points, axis=0)
    x1, y1 = np.max(points, axis=0)
    extent = max(x1-x0, y1-y0, 20)
    margin = extent * 0.12
    xlim = (x0-margin, x1+margin)
    ylim = (y1+margin, y0-margin)
    image = None
    if image_path and image_path.is_file():
        from PIL import Image
        with Image.open(image_path) as im:
            image = im.convert("RGB").copy()

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.7))
    if len(panels) == 1:
        axes = [axes]
    for ax, (subtitle, predicted, color) in zip(axes, panels):
        if image is not None:
            ax.imshow(image)
        draw_pose(ax, gt[:,:2], "tab:green", "GT (v > 0)", gt_vis)
        draw_pose(ax, predicted, color, subtitle)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        if image is None:
            ax.set_facecolor("#f7f7f7")
            ax.set_xlabel("Original-image x (px)")
            ax.set_ylabel("Original-image y (px)")
        else:
            ax.axis("off")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(title + ("" if image is not None else "  |  coordinate-only (no image)"), fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def get_top_two_logits(model, features, person_i, joint_i, axis_i):
    """Read current-run top-two SimCC scores; not the original NIE logits."""
    simcc = model.head.predict_from_joint_features(features)
    if not isinstance(simcc, (tuple, list)) or len(simcc) < 2:
        return {"current_simcc_logits_available": False}
    dist = simcc[axis_i]
    if hasattr(dist, "detach"):
        dist = dist.detach().cpu().numpy()
    dist = np.asarray(dist)
    if dist.ndim != 3 or dist.shape[0] <= person_i or dist.shape[1] <= joint_i:
        return {"current_simcc_logits_available": False}
    logits = dist[person_i, joint_i].astype(np.float64)
    top1 = int(np.argmax(logits))
    remainder = logits.copy()
    remainder[top1] = -np.inf
    top2 = int(np.argmax(remainder))
    indices = (top1, top2)
    return {
        "current_simcc_logits_available": True,
        "current_logits_top1_bin": int(indices[0]),
        "current_logits_top2_bin": int(indices[1]),
        "current_logits_top1_value": float(logits[indices[0]]),
        "current_logits_top2_value": float(logits[indices[1]]),
        "current_logits_top1_minus_top2": float(logits[indices[0]] - logits[indices[1]]),
    }


def verify_visual_cases(args, out_review: Path, chosen: list, outliers: list):
    # Load CIGPose only for a handful of full original batches. This keeps
    # inference identical in batch composition/order to label generation.
    import torch
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "mmpose"))
    sys.path.insert(0, str(repo / "mmpose/analysis/fyp"))
    from mmengine.config import Config
    from mmpose.apis import init_model
    from build_a_labels import decode_image, CKPT_SHA
    from oks_utility import compute_instance_oks

    if sha256(args.checkpoint) != CKPT_SHA:
        raise RuntimeError("Checkpoint SHA256 differs from original label-generation checkpoint")
    cfg = Config.fromfile(str(repo / "mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py"))
    cfg.model.test_cfg.flip_test = False
    model = init_model(cfg, str(args.checkpoint), device=args.device)
    model.eval().requires_grad_(False)

    wanted = set(r["annotation_id"] for r in chosen)
    annotation_data = read_json(args.cache / "selected_annotations.json")
    annotations = {int(a["id"]): a for a in annotation_data["annotations"] if int(a["id"]) in wanted}
    chosen_by_source = defaultdict(list)
    for r in chosen:
        chosen_by_source[r["source_shard"]].append(r)
    diagnostic_sources = [r["source_shard"] for r in outliers]
    source_names = sorted(set(chosen_by_source) | set(diagnostic_sources))
    p0_examples = []
    sem_cases = []
    p0_folder = out_review / "p0_outliers"
    gain_folder = out_review / "gain_cases"
    p0_folder.mkdir(exist_ok=True)
    gain_folder.mkdir(exist_ok=True)
    verified = 0
    failures = []
    for t, source_name in enumerate(source_names, 1):
        src_path = args.cache / source_name
        with np.load(src_path, allow_pickle=False) as z:
            orig = torch.from_numpy(z["features"].astype(np.float32)).to(args.device)
            p0_cached = z["P_0_coords"].astype(np.float64)
            center = z["input_center"].astype(np.float64)
            scale = z["input_scale"].astype(np.float64)
            size = z["input_size"].astype(np.float64)
            image_ids = z["image_id"].astype(np.int64)
            ann_ids = z["annotation_id"].astype(np.int64)
            original_size = np.asarray(z["original_size"])
        n = len(ann_ids)
        with torch.inference_mode():
            rebuilt = decode_image(model, orig, center, scale, size)
            # Alpha=0 semantic verification for every checked source batch.
            zero_joint = torch.zeros(n, dtype=torch.long, device=args.device)
            unchanged, zero_mask = model.head.cim.apply_intervention(
                orig.clone(), zero_joint, torch.zeros(n, device=args.device)
            )
            alpha0_feature_equal = bool(torch.equal(unchanged, orig) and not zero_mask.any())
            alpha0_pose_error = float(np.max(np.abs(decode_image(model, unchanged, center, scale, size) - rebuilt)))
            if not alpha0_feature_equal or alpha0_pose_error > 1e-4:
                failures.append(f"alpha=0 failed: {source_name}")

            if source_name in diagnostic_sources:
                absolute = np.abs(rebuilt - p0_cached)
                i, j, axis = [int(v) for v in np.unravel_index(np.argmax(absolute), absolute.shape)]
                aid = int(ann_ids[i])
                # Load annotation on demand for outlier shards outside selected Gain cases.
                if aid not in annotations:
                    selected = next(a for a in annotation_data["annotations"] if int(a["id"]) == aid)
                    annotations[aid] = selected
                ann = annotations[aid]
                gt = np.asarray(ann["keypoints"], dtype=np.float64).reshape(17, 3)
                area = float(ann["area"])
                oks_cached = float(compute_instance_oks(p0_cached[i], gt, area))
                oks_rebuilt = float(compute_instance_oks(rebuilt[i], gt, area))
                detail = {
                    "source_shard": source_name, "person_index": i,
                    "image_id": int(image_ids[i]), "annotation_id": aid,
                    "joint_id": j, "axis": "xy"[axis],
                    "gt_visibility_of_discrepant_joint": int(gt[j, 2]),
                    "cached_coordinate": float(p0_cached[i,j,axis]),
                    "rebuilt_coordinate": float(rebuilt[i,j,axis]),
                    "absolute_difference_px": float(absolute[i,j,axis]),
                    "oks_cached_p0": oks_cached,
                    "oks_rebuilt_p0": oks_rebuilt,
                    "absolute_oks_difference": abs(oks_cached-oks_rebuilt),
                    "alpha0_features_equal": alpha0_feature_equal,
                    "alpha0_vs_rebuilt_max_px": alpha0_pose_error,
                }
                try:
                    detail.update(get_top_two_logits(model, orig, i, j, axis))
                except Exception as exc:
                    detail["logits_diagnostic_error"] = str(exc)
                p0_examples.append(detail)
                if len([x for x in p0_examples if x.get("figure")]) < args.p0_figures:
                    png = p0_folder / f"p0_{source_name.replace('.npz','')}_ann{aid}.png"
                    path = args.images_dir / f"{int(image_ids[i]):012d}.jpg" if args.images_dir else None
                    draw_panels(png,
                                f"Cached NIE P0 vs rebuilt P0 | ann={aid} | max d={detail['absolute_difference_px']:.3f}px",
                                [("NIE cached P0", p0_cached[i], "tab:blue"),
                                 ("Rebuilt via shared decoder", rebuilt[i], "tab:purple")],
                                gt, original_size[i], path)
                    detail["figure"] = str(png.relative_to(out_review))

            for case in chosen_by_source.get(source_name, []):
                i = int(case["person_index_in_shard"])
                joint = int(case["joint_id"])
                aid = int(case["annotation_id"])
                if int(ann_ids[i]) != aid or int(image_ids[i]) != case["image_id"]:
                    raise RuntimeError(f"Source identity mismatch: {source_name}, {aid}")
                ann = annotations[aid]
                if int(ann["image_id"]) != case["image_id"]:
                    raise RuntimeError(f"GT image identity mismatch: {aid}")
                gt = np.asarray(ann["keypoints"], dtype=np.float64).reshape(17,3)
                original_snapshot = orig.clone()
                actions = torch.full((n,), joint, dtype=torch.long, device=args.device)
                candidate_features, mask = model.head.cim.apply_intervention(
                    orig.clone(), actions, torch.ones(n, device=args.device)
                )
                mask_ok = bool(mask.shape == (n, 17) and mask[:, joint].all()
                               and torch.count_nonzero(mask).item() == n)
                untouched = bool(torch.equal(orig, original_snapshot))
                candidate = decode_image(model, candidate_features, center, scale, size)
                recomputed_before = float(compute_instance_oks(p0_cached[i], gt, float(ann["area"])))
                recomputed_after = float(compute_instance_oks(candidate[i], gt, float(ann["area"])))
                recomputed_gain = recomputed_after - recomputed_before
                difference = abs(recomputed_after - case["oks_after_saved"])
                before_difference = abs(recomputed_before - case["oks_before_saved"])
                gain_difference = abs(recomputed_gain - case["gain_saved"])
                passed = bool(mask_ok and untouched and alpha0_feature_equal and alpha0_pose_error <= 1e-4
                              and max(difference, before_difference, gain_difference) <= 1e-6)
                if not passed:
                    failures.append(f"Semantic check mismatch for ann={aid}, joint={joint}: after diff={difference:.3g}")
                else:
                    verified += 1
                case.update({
                    "oks_before_recomputed": recomputed_before,
                    "oks_after_recomputed": recomputed_after,
                    "gain_recomputed": recomputed_gain,
                    "before_absolute_difference": before_difference,
                    "after_absolute_difference": difference,
                    "gain_absolute_difference": gain_difference,
                    "alpha0_vs_rebuilt_max_px": alpha0_pose_error,
                    "alpha0_features_equal": alpha0_feature_equal,
                    "alpha1_intervention_mask_ok": mask_ok,
                    "case_semantics_pass": passed,
                    "gt_visibility_of_selected_joint": int(gt[j,2]),
                })
                png = gain_folder / f"{case['case_type']}_ann{aid}_joint{joint}.png"
                path = args.images_dir / f"{int(image_ids[i]):012d}.jpg" if args.images_dir else None
                draw_panels(png,
                            f"{case['case_type'].upper()} | ann={aid} j={joint} | Gain={case['gain_saved']:+.5f} (OKS)",
                            [("NIE cached P0", p0_cached[i], "tab:blue"),
                             ("Rebuilt P0", rebuilt[i], "tab:purple"),
                             (f"Alpha1 candidate j={joint}", candidate[i], "tab:orange")],
                            gt, original_size[i], path)
                case["figure"] = str(png.relative_to(out_review))
        print(f"Reviewed source batch {t}/{len(source_names)}: {source_name}", flush=True)

    write_csv(out_review / "gain_cases.csv", chosen)
    write_csv(out_review / "p0_outlier_cases.csv", p0_examples)
    semantic = {
        "scope": "Selected sample batches only; not a full re-inference of 2,546,821 candidates",
        "selected_gain_candidates": len(chosen),
        "passed_gain_candidates": verified,
        "selected_p0_outlier_source_shards": len(diagnostic_sources),
        "alpha0_checks_per_tested_source_batch": len(source_names),
        "tolerance_oks": 1e-6,
        "issues": failures,
        "passed": len(failures) == 0 and verified == len(chosen),
    }
    write_json(out_review / "semantic_checks.json", semantic)
    return semantic, p0_examples


def make_markdown(dest: Path, p0: dict, gain_stats: dict, sem: dict, chosen: list, examples: list, out: Path):
    lines = [
        "# Module A | True Gain Labels: Supplementary Review",
        "",
        "## Provenance of P0 comparison",
        "",
        "**Official P0:** `P_0_coords` from NIE's COCO train2017 GT-box cache.",
        "",
        "**Rebuilt P0:** decode the original, *unmodified* `features` from the same NIE cache using TANG's shared `predict_from_joint_features` and MMPose `decode`, then restore original-image coordinates.",
        "",
        "This is **NOT** a comparison with any separately saved 'TANG P0' dataset.",
        "",
        "The official labels use **NIE cached P0** for `oks_before`; reconstructed P0 is for diagnostics only.",
        "",
        "## Full-data checks",
        "",
        "- COCO train2017 GT-box: **149,813 persons**, 4,682 shards, 17 candidates per person.",
        "- Independent data audit: `Full coverage=True`, `Issues=0` (see original `validation_report.json`).",
        "- The independent audit verifies schema/IDs/coverage/OKS-before/Gain arithmetic, not re-inference of all candidates.",
        "",
        "## Full Train2017 Gain statistics",
        "",
        f"- Total candidate gains: {gain_stats['total_candidate_gains']:,}",
        f"- Positive/negative/zero: {gain_stats['candidate_sign_counts']['positive']:,} / {gain_stats['candidate_sign_counts']['negative']:,} / {gain_stats['candidate_sign_counts']['zero']:,}",
        f"- Mean true Gain: {gain_stats['mean_gain']:+.8f}",
        f"- People with at least one positive candidate: {gain_stats['persons_with_at_least_one_positive_candidate']:,}",
        f"- Mean offline oracle-best Gain: {gain_stats['mean_oracle_best_gain']:+.8f} (not a learned-model result)",
        "",
        "## P0 difference statistics (all 4,682 logged shards)",
        "",
        f"- Weighted mean absolute coordinate difference: **{p0['mean_absolute_coordinate_difference_px']:.8f} px**",
        f"- Maximum absolute coordinate difference: **{p0['maximum_absolute_coordinate_difference_px']:.5f} px**",
        f"- Weighted mean absolute instance OKS difference: **{p0['mean_absolute_instance_oks_difference']:.9f}**",
        f"- Maximum absolute instance OKS difference: **{p0['maximum_absolute_instance_oks_difference']:.9f}**",
        "- Outlier threshold counts below refer to **shards with a large maximum difference**, not affected people or joints.",
    ]
    for threshold, num in p0["shards_with_max_coordinate_difference_above_px"].items():
        lines.append(f"  - max > {threshold} px: {num} source shards")
    lines += [
        "",
        "## Re-run semantic evidence",
        "",
        f"- Example candidates: {sem['selected_gain_candidates']}",
        f"- Recomputed candidate checks passed: {sem['passed_gain_candidates']}",
        f"- Candidate/alpha0 semantic status: **{'PASS' if sem['passed'] else 'NEEDS REVIEW'}**",
        "- Each example was re-inferred in its original full 32-person source batch; results were checked against saved OKS and Gain.",
        "- These sample checks do not replace a full model re-inference audit.",
        "",
        "## Unresolved technical question for NIE / TANG",
        "",
        "Does the original NIE cache P0 use exactly the same SimCC logit precision, argmax decoding, unmodified feature path, and original-image restoration as the reconstruction?",
        "Large outliers are not assumed to be adjacent-bin effects. The saved cache does not include the original NIE SimCC logits, so the exact cause cannot be proven from rebuilt logits alone.",
        "",
        "## Artifacts",
        "",
        "- `true_gain_summary.json`, `per_joint_gain.csv`, `true_gain_distribution.png`: full-data Gain statistics.",
        "- `p0_difference_summary.json`, `p0_shard_max_*.png`: full-data shard-level difference statistics and plots.",
        "- `p0_top_shards.csv`, `p0_top_oks_shards.csv`: shard-level outliers.",
        "- `p0_outlier_cases.csv`, `p0_outliers/*.png`: individual largest-difference cases.",
        "- `gain_cases.csv`, `gain_cases/*.png`: positive/negative gain samples; candidate predictions rechecked.",
        "- `semantic_checks.json`: exact sample validation results and any issues.",
        "",
        "### Image background",
        "",
        "If COCO train2017 image files are not supplied, figures show original-image coordinate plots on a neutral background; they are not photographic overlays.",
        "",
    ]
    (dest / "REVIEW_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Existing full train_a_labels output directory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--images-dir", type=Path, default=None, help="Optional folder containing COCO 000000000000.jpg images")
    ap.add_argument("--examples-per-sign", type=int, default=3)
    ap.add_argument("--p0-sources", type=int, default=12, help="Full batches to reconstruct for P0 outlier analysis")
    ap.add_argument("--p0-figures", type=int, default=6)
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.images_dir:
        args.images_dir = args.images_dir.resolve()
        if not args.images_dir.is_dir():
            ap.error(f"COCO images directory not found: {args.images_dir}")
    if not args.cache.is_dir() or not args.checkpoint.is_file() or not args.repo.is_dir():
        ap.error("Repo, cache, and checkpoint must exist")
    if not args.output.is_dir():
        ap.error(f"Label directory not found: {args.output}")
    if args.examples_per_sign < 1 or args.p0_sources < 1 or args.p0_figures < 0:
        ap.error("All case counts must be nonnegative (examples/sources >= 1)")
    dest = args.output / "review_a_labels"
    dest.mkdir(exist_ok=True)
    print("=== MODULE A FULL LABEL REVIEW ===", flush=True)
    status = read_json(args.output / "status.json")
    report = read_json(args.output / "validation_report.json")
    p0, p0_ranked, oks_ranked = aggregate_p0(args.output, status, report, dest)
    print(f"P0 diagnostics aggregated for {p0['source_shards']} shards", flush=True)
    full, blocks, boundaries = load_labels_for_cases(args.output, status)
    gain_stats = summarize_true_gains(full, dest)
    chosen = choose_examples(full, blocks, boundaries, args.examples_per_sign)
    print(f"Selected {len(chosen)} Gain cases; rerunning source batches", flush=True)
    outliers = p0_ranked[:args.p0_sources]
    sem, examples = verify_visual_cases(args, dest, chosen, outliers)
    make_markdown(dest, p0, gain_stats, sem, chosen, examples, args.output)
    print(f"P0 max coordinate error: {p0['maximum_absolute_coordinate_difference_px']:.5f} px", flush=True)
    print(f"P0 max OKS difference: {p0['maximum_absolute_instance_oks_difference']:.9f}", flush=True)
    print(f"Gain sample semantics: {sem['passed_gain_candidates']}/{sem['selected_gain_candidates']}", flush=True)
    print("Supplementary review:", dest, flush=True)
    if not sem["passed"]:
        raise SystemExit("Some re-inferred cases need review; see semantic_checks.json")
    print("PASS: Supplementary review created", flush=True)


if __name__ == "__main__":
    main()
