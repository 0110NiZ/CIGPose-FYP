
"""Step 2D: Model interface and full-validation OKS audit.

Usage:
    python step2d_validation.py --baseline-dir PATH --model-python PYTHON

The existing interface tests expect COCO at <repo>/data/coco and
the checkpoint at <repo>/checkpoints/cigpose-m_coco_256x192.pth.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]

SIGMAS = np.array([
    .26, .25, .25, .35, .35,
    .79, .79, .72, .72, .62, .62,
    1.07, 1.07, .87, .87, .89, .89
], dtype=np.float64) / 10.0


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig")
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def git_head():
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True
    )
    return (
        result.stdout.strip()
        if result.returncode == 0
        else "unavailable"
    )


def load_lin_oks(lin_file):
    if lin_file:
        source = Path(lin_file).read_text(
            encoding="utf-8"
        )
    else:
        local = (
            REPO / "mmpose/analysis/fyp/oks_utility.py"
        )

        if local.is_file():
            source = local.read_text(encoding="utf-8")
        else:
            result = subprocess.run(
                [
                    "git", "-C", str(REPO), "show",
                    "origin/lin:mmpose/analysis/fyp/"
                    "oks_utility.py"
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise FileNotFoundError(
                    "LIN OKS utility missing. "
                    "Use --lin-file or fetch the lin branch."
                )

            source = result.stdout

    module = types.ModuleType("lin_oks_utility")
    exec(
        compile(source, "oks_utility.py", "exec"),
        module.__dict__
    )

    return module, hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()


def independent_oks(prediction, ground_truth, area):
    pred = np.asarray(
        prediction, dtype=np.float64
    )
    gt = np.asarray(
        ground_truth, dtype=np.float64
    )

    if pred.shape != (17, 2) or gt.shape != (17, 3):
        raise ValueError("Incorrect keypoint shape")

    if (
        not np.isfinite(pred).all()
        or not np.isfinite(area)
        or area <= 0
    ):
        raise ValueError(
            "Invalid predicted coordinates or GT area"
        )

    visible = gt[:, 2] > 0

    if (
        not visible.any()
        or not np.isin(gt[:, 2], [0, 1, 2]).all()
    ):
        raise ValueError("Invalid GT visibility")

    if not np.isfinite(gt[visible, :2]).all():
        raise ValueError("Invalid labeled GT coordinates")

    dx = pred[visible, 0] - gt[visible, 0]
    dy = pred[visible, 1] - gt[visible, 1]

    e = (
        (dx ** 2 + dy ** 2)
        / ((2 * SIGMAS[visible]) ** 2)
        / (area + np.spacing(1))
        / 2
    )

    return float(np.exp(-e).mean())


def run_interface_tests(model_python, output):
    mm_root = REPO / "mmpose"

    scripts = [
        "check_intervention_interface.py",
        "check_topdown_alignment.py"
    ]

    required = [
        REPO / "data/coco",
        REPO / "checkpoints/"
        "cigpose-m_coco_256x192.pth"
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Model test prerequisite missing: {path}"
            )

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(mm_root)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    results = []

    log_names = [
        "interface_test.log",
        "alignment_test.log"
    ]

    for name, log_name in zip(scripts, log_names):
        script = mm_root / "analysis/fyp" / name

        if not script.is_file():
            raise FileNotFoundError(script)

        result = subprocess.run(
            [str(model_python), "-u", str(script)],
            cwd=str(mm_root),
            env=env,
            capture_output=True,
            text=True
        )

        (output / log_name).write_text(
            result.stdout + "\n" + result.stderr,
            encoding="utf-8"
        )

        passed = (
            result.returncode == 0
            and "OVERALL: PASS" in result.stdout
        )

        results.append({
            "test": name,
            "passed": passed
        })

        print(name, "PASS" if passed else "FAIL")

    return results


def audit_full_predictions(baseline_dir, lin):
    directory = Path(baseline_dir)

    gt_path = directory / (
        "person_keypoints_val2017.json"
    )

    gt = load_json(gt_path)

    manifest = load_json(
        directory / "val2017_valid_persons.json"
    )

    p0 = load_json(
        directory
        / "p0/raw_predictions_before_nms.json"
    )

    pc = load_json(
        directory
        / "top1_alpha1/raw_predictions_before_nms.json"
    )

    errors = []
    rows = []

    valid_ids = [
        int(a) for a in manifest["valid_ann_ids"]
    ]
    valid_set = set(valid_ids)

    gt_map = {
        int(a["id"]): a
        for a in gt["annotations"]
    }

    def index(items, name):
        ids = [
            int(p["annotation_id"])
            for p in items
        ]

        if (
            len(ids) != len(set(ids))
            or set(ids) != valid_set
        ):
            errors.append(
                f"{name}: duplicate, missing, or "
                "unexpected annotation IDs"
            )

        return {
            int(p["annotation_id"]): p
            for p in items
        }

    p0_map = index(p0, "P0")
    pc_map = index(pc, "Pc")

    if (
        len(valid_ids) != 6352
        or len(valid_ids) != len(valid_set)
    ):
        errors.append(
            "Invalid valid-person manifest count/uniqueness"
        )

    if len(gt_map) != len(gt["annotations"]):
        errors.append("Duplicate GT annotation IDs")

    hash_ok = (
        sha256(gt_path)
        == manifest.get("source_annotation_sha256")
    )

    if not hash_ok:
        errors.append("GT SHA256 mismatch")

    max_diff = 0.0

    for aid in sorted(valid_set):
        try:
            ann = gt_map[aid]
            p0_pred = p0_map[aid]
            pc_pred = pc_map[aid]

            image_id = int(ann["image_id"])

            if (
                int(p0_pred["image_id"]) != image_id
                or int(pc_pred["image_id"]) != image_id
            ):
                raise ValueError(
                    "Image ID does not match GT"
                )

            if (
                abs(float(p0_pred["alpha"])) > 1e-8
                or abs(float(pc_pred["alpha"]) - 1) > 1e-8
            ):
                raise ValueError("Alpha mismatch")

            joint_id = int(pc_pred["joint_id"])

            if not 0 <= joint_id < 17:
                raise ValueError("Invalid joint ID")

            gt_points = np.asarray(
                ann["keypoints"],
                dtype=np.float64
            ).reshape(17, 3)

            area = float(ann["area"])

            p0_coords = np.asarray(
                p0_pred["keypoints"],
                dtype=np.float64
            )

            pc_coords = np.asarray(
                pc_pred["keypoints"],
                dtype=np.float64
            )

            oks_p0 = independent_oks(
                p0_coords, gt_points, area
            )

            oks_pc = independent_oks(
                pc_coords, gt_points, area
            )

            gain = oks_pc - oks_p0

            difference = max(
                abs(
                    oks_p0
                    - lin.compute_instance_oks(
                        p0_coords, gt_points, area
                    )
                ),
                abs(
                    oks_pc
                    - lin.compute_instance_oks(
                        pc_coords, gt_points, area
                    )
                ),
                abs(
                    gain
                    - lin.compute_gain(
                        pc_coords,
                        p0_coords,
                        gt_points,
                        area
                    )
                )
            )

            max_diff = max(
                max_diff, difference
            )

            if difference > 1e-10:
                raise ValueError(
                    f"LIN OKS disagreement: {difference}"
                )

            rows.append({
                "annotation_id": aid,
                "image_id": image_id,
                "joint_id": joint_id,
                "oks_p0": oks_p0,
                "oks_pc": oks_pc,
                "gain": gain
            })

        except Exception as exc:
            errors.append(
                f"annotation_id={aid}: {exc}"
            )

    if len(rows) != len(valid_set):
        errors.append("Incomplete validation")

    gains = np.asarray(
        [r["gain"] for r in rows]
    )

    report = {
        "expected_persons": len(valid_set),
        "validated_persons": len(rows),
        "unique_annotation_ids": len({
            r["annotation_id"] for r in rows
        }),
        "gt_sha256_match": hash_ok,
        "lin_comparison": (
            "PASS" if max_diff <= 1e-10 else "FAIL"
        ),
        "max_difference_vs_lin": max_diff,
        "mean_oks_p0": (
            float(np.mean([
                r["oks_p0"] for r in rows
            ])) if rows else None
        ),
        "mean_oks_pc": (
            float(np.mean([
                r["oks_pc"] for r in rows
            ])) if rows else None
        ),
        "mean_gain": (
            float(gains.mean()) if len(gains) else None
        ),
        "positive_gain_count": int(
            np.sum(gains > 1e-10)
        ),
        "negative_gain_count": int(
            np.sum(gains < -1e-10)
        ),
        "near_zero_gain_count": int(
            np.sum(np.abs(gains) <= 1e-10)
        )
    }

    return rows, report, errors


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--model-python",
        default=sys.executable,
        help="Python with CIGPose/MMCV installed"
    )

    parser.add_argument(
        "--lin-file",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            REPO / "validation_results/step2d"
        )
    )

    parser.add_argument(
        "--skip-model-tests",
        action="store_true"
    )

    args = parser.parse_args()
    args.out_dir.mkdir(
        parents=True, exist_ok=True
    )

    errors = []

    report = {
        "task": "Step 2D Full Validation",
        "git_commit": git_head(),
        "overall": "FAIL"
    }

    try:
        if args.skip_model_tests:
            tests = []
        else:
            tests = run_interface_tests(
                args.model_python,
                args.out_dir
            )

            if not all(
                test["passed"] for test in tests
            ):
                errors.append(
                    "Model interface/alignment test failed"
                )

        lin, lin_hash = load_lin_oks(
            args.lin_file
        )

        rows, summary, calculation_errors = (
            audit_full_predictions(
                args.baseline_dir, lin
            )
        )

        errors.extend(calculation_errors)

        report.update(summary)

        report.update({
            "interface_tests": tests,
            "lin_source_sha256": lin_hash
        })

        csv_path = (
            args.out_dir
            / "step2d_person_oks_gain.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "annotation_id",
                    "image_id",
                    "joint_id",
                    "oks_p0",
                    "oks_pc",
                    "gain"
                ]
            )

            writer.writeheader()
            writer.writerows(rows)

        if errors:
            report["overall"] = "FAIL"
        elif args.skip_model_tests:
            report["overall"] = "PARTIAL"
        else:
            report["overall"] = "PASS"

    except Exception as exc:
        errors.append(
            f"{type(exc).__name__}: {exc}"
        )

    report["error_count"] = len(errors)
    report["errors"] = errors[:50]

    report_path = (
        args.out_dir / "step2d_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    (
        args.out_dir / "step2d_run.log"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSTEP 2D FINAL SUMMARY")

    for key, value in report.items():
        if key not in (
            "errors",
            "git_commit",
            "lin_source_sha256"
        ):
            print(f"{key}: {value}")

    print("Results:", args.out_dir)

    return (
        0 if report["overall"] == "PASS" else 1
    )


if __name__ == "__main__":
    sys.exit(main())
