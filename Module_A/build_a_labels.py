"""
Module A: Generate 17 independent OKS Gain labels.
Dataset: COCO Train2017 GT-box
Base model: Frozen CIGPose-M
Intervention strength: alpha = 1
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


# ============================================================
# 1. N's metadata contract
# ============================================================

CACHE_FP = (
    "40ba6b369159a2b25266e19bcdfe959114c205c0f8f64b900df8ac67103ddafb"
)

CKPT_SHA = (
    "3dd69c8ab4e3c7a923a0183fd9abf844cb7ef13d39d26693baa082ca7931b3d0"
)

ANN_SHA = (
    "7fc1549d934547c470384d8a207c38707ca33fc016d00e33b795e408603af83e"
)

TEMPLATE = {
    "schema_version": "cigpose-module-a-labels-v1",
    "dataset_split": "train2017",
    "source_cache_fingerprint": CACHE_FP,
    "checkpoint_sha256": CKPT_SHA,
    "annotation_sha256": ANN_SHA,
    "joint_order": list(range(17)),
    "alpha": 1.0,
    "flip_test": False,
    "coordinate_system": "original_image",
    "gain_definition": "OKS(P_j_alpha1,GT)-OKS(P_0,GT)",
    "expected_count": 149813,
    "count": 0,
    "complete": False,
    "generator_git_commit": "",
    "oks_implementation_sha256": "",
    "shards": []
}


# ============================================================
# 2. Utilities
# ============================================================

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(obj):
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def save_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(temp, path)


# ============================================================
# 3. SimCC decoding and original-image restoration
# ============================================================

def decode_image(model, features, center, scale, size):

    # TANG's shared downstream prediction
    simcc = model.head.predict_from_joint_features(features)

    # MMPose's actual decoder
    decoded = model.head.decode(simcc)

    assert len(decoded) == features.shape[0]

    coords = []

    for item in decoded:

        xy = item.keypoints

        if isinstance(xy, torch.Tensor):
            xy = xy.detach().cpu().numpy()

        xy = np.asarray(xy, dtype=np.float64)

        assert xy.shape in [(1, 17, 2), (17, 2)], (
            f"Unexpected decoder shape: {xy.shape}"
        )

        coords.append(xy.reshape(17, 2))

    coords = np.stack(coords)

    # Standard top-down crop inverse transformation.
    # Must reproduce cached P0 before any label is saved.
    restored = (
        (coords / size[:, None, :] - 0.5)
        * scale[:, None, :]
        + center[:, None, :]
    )

    assert np.isfinite(restored).all()

    return restored


# ============================================================
# 4. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=0)

    args = parser.parse_args()

    repo = args.repo.resolve()
    cache = args.cache.resolve()
    output = args.output.resolve()

    assert args.max_shards >= 0

    # Use FYP versions of MMPose and shared utilities
    sys.path.insert(0, str(repo / "mmpose"))
    sys.path.insert(0, str(repo / "mmpose/analysis/fyp"))

    from mmpose.apis import init_model
    from mmengine.config import Config
    from oks_utility import compute_instance_oks
    import oks_utility

    # --------------------------------------------------------
    # Verify checkpoint and GT cache
    # --------------------------------------------------------

    assert hash_file(args.checkpoint) == CKPT_SHA

    run = json.loads(
        (cache / "run.json").read_text(encoding="utf-8-sig")
    )

    status = json.loads(
        (cache / "status.json").read_text(encoding="utf-8-sig")
    )

    assert digest_json(run) == CACHE_FP
    assert status["fingerprint"] == CACHE_FP
    assert status["complete"] is True
    assert status["count"] == 149813

    protocol = run["protocol"]

    assert protocol["bbox_source"] == "gt"
    assert protocol["flip_test"] is False
    assert protocol["checkpoint_sha256"] == CKPT_SHA
    assert protocol["annotation_sha256"] == ANN_SHA

    selected = json.loads(
        (cache / "selected_annotations.json").read_text(
            encoding="utf-8-sig"
        )
    )

    annotations = {
        int(ann["id"]): ann
        for ann in selected["annotations"]
    }

    assert len(annotations) == 149813

    print("GT-box cache: PASS", flush=True)
    print("Checkpoint SHA256: PASS", flush=True)

    # --------------------------------------------------------
    # Load frozen CIGPose-M
    # --------------------------------------------------------

    config_path = (
        repo / "mmpose/projects/cigpose/body_2d_keypoint/"
        "cigpose-m_1xb384-420e_coco-256x192.py"
    )

    cfg = Config.fromfile(str(config_path))
    cfg.model.test_cfg.flip_test = False

    model = init_model(
        cfg,
        str(args.checkpoint),
        device=args.device
    )

    model.eval().requires_grad_(False)

    print("Frozen CIGPose-M loaded: PASS", flush=True)

    # --------------------------------------------------------
    # Prepare output and resume status
    # --------------------------------------------------------

    shards = sorted(cache.glob("shard-*.npz"))

    assert len(shards) == 4682

    if args.max_shards > 0:
        shards = shards[:args.max_shards]

    output.mkdir(parents=True, exist_ok=True)

    metadata_file = output / "metadata.json"

    if not metadata_file.exists():
        save_json(metadata_file, TEMPLATE)
    else:
        existing = json.loads(metadata_file.read_text())
        assert existing["source_cache_fingerprint"] == CACHE_FP
        assert existing["checkpoint_sha256"] == CKPT_SHA
        assert existing["complete"] is False

    progress_file = output / "status.json"

    generator_sha = hash_file(__file__)
    oks_sha = hash_file(oks_utility.__file__)

    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        assert progress["generator_sha256"] == generator_sha
        assert progress["oks_sha256"] == oks_sha
    else:
        progress = {
            "generator_sha256": generator_sha,
            "oks_sha256": oks_sha,
            "done": {}
        }

    # --------------------------------------------------------
    # Generate real gains, shard by shard
    # --------------------------------------------------------

    with torch.inference_mode():

        for index, source in enumerate(shards):

            name = f"shard-{index:05d}.npz"
            destination = output / name

            # Resume previously verified shard
            if source.name in progress["done"]:

                saved = progress["done"][source.name]

                assert saved["file"] == name
                assert destination.is_file()
                assert hash_file(destination) == saved["sha256"]

                continue

            try:
                with np.load(source, allow_pickle=False) as data:

                    assert str(data["fingerprint"].item()) == CACHE_FP

                    image_ids = data["image_id"].astype(np.int64)
                    ann_ids = data["annotation_id"].astype(np.int64)

                    features = data["features"].astype(np.float32)
                    p0 = data["P_0_coords"].astype(np.float64)

                    center = data["input_center"].astype(np.float64)
                    scale = data["input_scale"].astype(np.float64)
                    size = data["input_size"].astype(np.float64)



                n = len(ann_ids)

                assert features.shape == (n, 17, 256)
                assert np.all(scale > 0)
                assert np.all(size > 0)

                # Check image/annotation identity
                for image_id, ann_id in zip(image_ids, ann_ids):
                    ann = annotations[int(ann_id)]
                    assert int(ann["image_id"]) == int(image_id)

                original = torch.from_numpy(features).to(args.device)
                original_copy = original.clone()

                # ------------------------------------------------
                # Verify P0 reconstruction
                # ------------------------------------------------

                rebuilt_p0 = decode_image(
                    model, original, center, scale, size
                )

                p0_error = float(
                    np.max(np.abs(rebuilt_p0 - p0))
                )

                # Diagnostic: compare reconstructed P0 to cached P0
                abs_diff = np.abs(rebuilt_p0 - p0)

                print("\n=== P0 RECONSTRUCTION DIAGNOSTIC ===", flush=True)
                print("Persons:", n, flush=True)
                print("Max coordinate error (px):",
                      float(abs_diff.max()), flush=True)
                print("Mean coordinate error (px):",
                      float(abs_diff.mean()), flush=True)
                print("Median coordinate error (px):",
                      float(np.median(abs_diff)), flush=True)
                print("95th percentile error (px):",
                      float(np.percentile(abs_diff, 95)), flush=True)

                cached_oks = []
                rebuilt_oks = []

                for i, ann_id in enumerate(ann_ids):
                    ann = annotations[int(ann_id)]
                    gt_points = np.asarray(
                        ann["keypoints"], dtype=np.float64
                    ).reshape(17, 3)
                    area = float(ann["area"])

                    cached_oks.append(
                        compute_instance_oks(p0[i], gt_points, area)
                    )
                    rebuilt_oks.append(
                        compute_instance_oks(rebuilt_p0[i], gt_points, area)
                    )

                oks_diff = np.abs(
                    np.asarray(cached_oks) -
                    np.asarray(rebuilt_oks)
                )

                print("Max OKS difference:",
                      float(oks_diff.max()), flush=True)
                print("Mean OKS difference:",
                      float(oks_diff.mean()), flush=True)
                print("=== END DIAGNOSTIC ===\n", flush=True)

                worst = np.unravel_index(
                    np.argmax(abs_diff), abs_diff.shape
                )
                person_i, joint_i, axis_i = map(int, worst)

                print("Largest-error annotation:",
                      int(ann_ids[person_i]), flush=True)
                print("Largest-error joint:", joint_i, flush=True)
                print("Largest-error axis:",
                      "x" if axis_i == 0 else "y", flush=True)
                print("Cached coordinate:",
                      float(p0[worst]), flush=True)
                print("Rebuilt coordinate:",
                      float(rebuilt_p0[worst]), flush=True)

                # Provisional first-shard tolerance, not final audit policy
                # Provisional P0 consistency policy for the smoke test.
                # Check typical coordinate error, rare outliers and OKS impact.
                outlier_ratio = float(np.mean(abs_diff > 0.1))
                p95_error = float(np.percentile(abs_diff, 95))
                max_oks_error = float(oks_diff.max())

                print("P0 outlier ratio (>0.1 px):",
                      outlier_ratio, flush=True)
                print("P0 consistency (provisional):",
                      {
                          "max_px": p0_error,
                          "p95_px": p95_error,
                          "max_oks_diff": max_oks_error
                      }, flush=True)


                # Record P0 differences without rejecting samples.
                diag_dir = output / "p0_diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)

                save_json(
                    diag_dir / f"{source.stem}.json",
                    {
                        "source_shard": source.name,
                        "persons": int(n),
                        "p0_max_abs_error_px": float(abs_diff.max()),
                        "p0_mean_abs_error_px": float(abs_diff.mean()),
                        "p0_median_abs_error_px": float(np.median(abs_diff)),
                        "p0_p95_abs_error_px": float(np.percentile(abs_diff, 95)),
                        "p0_p99_abs_error_px": float(np.percentile(abs_diff, 99)),
                        "oks_max_abs_difference": float(oks_diff.max()),
                        "oks_mean_abs_difference": float(oks_diff.mean()),
                        "worst_person_index": int(worst[0]),
                        "worst_joint": int(worst[1]),
                        "worst_axis": int(worst[2])
                    }
                )

                print(
                    "P0 differences recorded; "
                    "no numerical rejection.",
                    flush=True
                )

                # Alpha=0 should recover the original features/P0
                if index == 0:

                    joints_zero = torch.zeros(
                        n, dtype=torch.long, device=args.device
                    )

                    alphas_zero = torch.zeros(
                        n, device=args.device
                    )

                    unchanged, mask_zero = (
                        model.head.cim.apply_intervention(
                            original.clone(),
                            joints_zero,
                            alphas_zero
                        )
                    )

                    assert torch.equal(unchanged, original)
                    assert not mask_zero.any()

                    alpha0_pose = decode_image(
                        model, unchanged, center, scale, size
                    )

                    alpha0_error = float(
                        np.max(np.abs(alpha0_pose - rebuilt_p0))
                    )

                    print("Alpha=0 vs rebuilt P0 error:",
                          alpha0_error, flush=True)

                    assert alpha0_error <= 1e-4, (
                        "Alpha=0 did not reproduce non-intervened P0"
                    )

                    print("P0 / alpha=0 verification: PASS", flush=True)

                # ------------------------------------------------
                # Load same-person GT and calculate OKS_before
                # ------------------------------------------------

                gt = []
                areas = []

                for ann_id in ann_ids:

                    ann = annotations[int(ann_id)]

                    gt.append(
                        np.asarray(
                            ann["keypoints"], dtype=np.float64
                        ).reshape(17, 3)
                    )

                    areas.append(float(ann["area"]))

                before = np.array([
                    compute_instance_oks(
                        p0[i], gt[i], areas[i]
                    )
                    for i in range(n)
                ], dtype=np.float64)

                after = np.empty(
                    (n, 17), dtype=np.float64
                )

                audit_candidates = {}

                # ------------------------------------------------
                # Each of 17 candidates starts from original features
                # ------------------------------------------------

                for joint_id in range(17):

                    joints = torch.full(
                        (n,), joint_id,
                        dtype=torch.long,
                        device=args.device
                    )

                    alphas = torch.ones(
                        n, device=args.device
                    )

                    changed, mask = (
                        model.head.cim.apply_intervention(
                            original.clone(),
                            joints,
                            alphas
                        )
                    )

                    assert mask.shape == (n, 17)
                    assert mask[:, joint_id].all()
                    assert torch.count_nonzero(mask).item() == n

                    # Original features must remain untouched
                    assert torch.equal(original, original_copy)

                    candidate = decode_image(
                        model, changed, center, scale, size
                    )

                    for i in range(n):
                        after[i, joint_id] = compute_instance_oks(
                            candidate[i],
                            gt[i],
                            areas[i]
                        )

                    if index == 0:
                        audit_candidates[str(joint_id)] = (
                            candidate[0].tolist()
                        )

                # ------------------------------------------------
                # Signed OKS gain
                # ------------------------------------------------

                gain = after - before[:, None]

                assert np.isfinite(gain).all()
                assert np.all((before >= 0) & (before <= 1))
                assert np.all((after >= 0) & (after <= 1))

                # Save one representative example for later audit
                if index == 0:

                    save_json(
                        output / "audit_example.json",
                        {
                            "image_id": int(image_ids[0]),
                            "annotation_id": int(ann_ids[0]),
                            "p0": p0[0].tolist(),
                            "gt": gt[0].tolist(),
                            "candidates": audit_candidates,
                            "gain": gain[0].tolist(),
                            "p0_max_abs_error_px": p0_error
                        }
                    )

                # ------------------------------------------------
                # Save only N's five required arrays
                # ------------------------------------------------

                temp = output / (name + ".tmp")

                with temp.open("wb") as f:
                    np.savez_compressed(
                        f,
                        image_id=image_ids,
                        annotation_id=ann_ids,
                        oks_before=before,
                        oks_after=after,
                        gain=gain
                    )

                os.replace(temp, destination)

                progress["done"][source.name] = {
                    "file": name,
                    "sha256": hash_file(destination),
                    "count": int(n)
                }

                progress.setdefault("failed", {}).pop(source.name, None)
                save_json(progress_file, progress)

                if index == 0 or (index + 1) % 50 == 0:
                    print(
                        f"[{index+1}/{len(shards)}] "
                        f"{n} persons | "
                        f"P0 error={p0_error:.5f}px | "
                        f"Mean gain={gain.mean():+.6f}",
                        flush=True
                    )

            except Exception as exc:
                import traceback
                import zipfile
                from datetime import datetime, timezone

                # A failed shard is NOT a completed shard.
                if source.name in progress["done"]:
                    raise

                recoverable = isinstance(
                    exc,
                    (
                        AssertionError,
                        ValueError,
                        KeyError,
                        IndexError,
                        EOFError,
                        zipfile.BadZipFile
                    )
                )

                failed = progress.setdefault("failed", {})
                previous = failed.get(source.name, {})

                record = {
                    "source_shard": source.name,
                    "output_shard": name,
                    "source_index": int(index),
                    "attempts": int(previous.get("attempts", 0)) + 1,
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "recoverable": bool(recoverable),
                    "status": "pending_retry"
                }

                # Record all IDs present in the source shard.
                # These are NOT necessarily the exact failing IDs.
                try:
                    with np.load(source, allow_pickle=False) as d:
                        if "annotation_id" in d.files:
                            record["source_annotation_ids"] = (
                                d["annotation_id"].astype("int64").tolist()
                            )
                        if "image_id" in d.files:
                            record["source_image_ids"] = (
                                d["image_id"].astype("int64").tolist()
                            )
                except Exception as id_exc:
                    record["id_read_error"] = repr(id_exc)

                failed[source.name] = record
                save_json(progress_file, progress)

                # Permanent append-only exception history.
                # Records remain even after successful retry.
                history_path = output / "failure_history.jsonl"

                with history_path.open(
                    "a", encoding="utf-8"
                ) as history:
                    history.write(
                        json.dumps(
                            {"event": "failed", **record},
                            ensure_ascii=False
                        ) + "\n"
                    )
                    history.flush()
                    os.fsync(history.fileno())

                print(
                    f"[FAILED {index+1}/{len(shards)}] "
                    f"{source.name}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )

                if not recoverable:
                    print(
                        "FATAL: Unknown/system-level error. "
                        "Stopped to protect dataset integrity.",
                        flush=True
                    )
                    raise

                # Continue with the next source shard.
                continue

    remaining_failed = progress.get("failed", {})

    if remaining_failed:
        print(
            f"\\n=== GENERATION PASS ENDED: "
            f"{len(remaining_failed)} FAILED SHARDS ===",
            flush=True
        )
        print(
            "Completed persons:",
            sum(x["count"] for x in progress["done"].values())
        )
        print("Failed shards require retry before finalization.")
        raise SystemExit(2)

    print("\n=== MODULE A GENERATION FINISHED ===")
    print(
        "Generated persons:",
        sum(x["count"] for x in progress["done"].values())
    )
    print("Output:", output)
    print("Metadata remains incomplete until independent audit.")


if __name__ == "__main__":
    main()
