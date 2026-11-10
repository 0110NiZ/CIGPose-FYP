import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval

from oks_utility import COCO_SIGMAS, compute_instance_oks, compute_gain


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from fyp_data_common import (check_outputs, hash_metadata, read_json,
                             verify_manifest, write_json)

ATOL = 1e-12


def official_oks(pred, ann):
    """Call the actual COCOeval OKS method for one fixed person-GT pair."""
    evaluator = COCOeval(iouType="keypoints")
    image_id = ann["image_id"]
    category_id = ann["category_id"]

    # Prediction confidence is separate from GT visibility.
    pred_with_scores = np.column_stack(
        [pred, np.ones(17, dtype=np.float64)]
    )
    evaluator._gts[(image_id, category_id)] = [copy.deepcopy(ann)]
    evaluator._dts[(image_id, category_id)] = [{
        "image_id": image_id,
        "category_id": category_id,
        "keypoints": pred_with_scores.reshape(-1).tolist(),
        "score": 1.0,
    }]
    return float(evaluator.computeOks(image_id, category_id)[0, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--split", type=Path, default=ROOT / "splits/train_dev_split.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/train_dev_valid_ann_ids_v2.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/oks")
    parser.add_argument("--overwrite", action="store_true", help="Replace this validation report")
    args = parser.parse_args()
    report_path = args.out_dir / "oks_validation.json"
    if report_path.resolve() in {args.ann.resolve(), args.split.resolve(), args.manifest.resolve()}:
        parser.error("Output must not replace an input")
    check_outputs([report_path], args.overwrite)

    passed = []
    comparisons = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        passed.append(name)
        print("PASS:", name)

    def compare(name, pred, ann):
        gt = np.asarray(ann["keypoints"], dtype=np.float64).reshape(17, 3)
        ours = compute_instance_oks(pred, gt, ann["area"])
        official = official_oks(pred, ann)
        np.testing.assert_allclose(
            ours, official, rtol=0, atol=ATOL, err_msg=name
        )
        comparisons.append({
            "case": name,
            "annotation_id": ann["id"],
            "ours": ours,
            "official": official,
            "absolute_difference": abs(ours - official),
        })
        print("MATCH:", name)
        return ours

    # Check that joint-specific sigmas match the installed COCO reference.
    reference_sigmas = COCOeval(iouType="keypoints").params.kpt_oks_sigmas
    check(
        "COCO sigmas match",
        np.array_equal(COCO_SIGMAS, reference_sigmas),
    )

    gt = np.tile([100.0, 100.0, 2.0], (17, 1))
    perfect = gt[:, :2].copy()
    synthetic = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": [0, 0, 200, 200],
        "area": 10000.0,
        "num_keypoints": 17,
        "keypoints": gt.reshape(-1).tolist(),
    }

    compare("synthetic_perfect", perfect, synthetic)
    compare("synthetic_shifted", perfect + 10, synthetic)

    # v=1 must be included just like v=2.
    gt_occluded = gt.copy()
    gt_occluded[0, 2] = 1
    shifted_one = perfect.copy()
    shifted_one[0, 0] += 20
    check(
        "v=1 is included",
        compute_instance_oks(shifted_one, gt_occluded, 10000)
        == compute_instance_oks(shifted_one, gt, 10000)
        < 1.0,
    )
    occluded_ann = copy.deepcopy(synthetic)
    occluded_ann["keypoints"] = gt_occluded.reshape(-1).tolist()
    compare("synthetic_occluded", shifted_one, occluded_ann)

    # Changing an unannotated joint must not change OKS.
    gt_unlabeled = gt.copy()
    gt_unlabeled[0, 2] = 0
    check(
        "v=0 is ignored",
        compute_instance_oks(shifted_one, gt_unlabeled, 10000) == 1.0,
    )
    unlabeled_ann = copy.deepcopy(synthetic)
    unlabeled_ann["num_keypoints"] = 16
    unlabeled_ann["keypoints"] = gt_unlabeled.reshape(-1).tolist()
    compare("synthetic_unlabeled", shifted_one, unlabeled_ann)

    check(
        "larger error reduces OKS",
        compute_instance_oks(perfect + 20, gt, 10000)
        < compute_instance_oks(perfect + 10, gt, 10000)
        < 1.0,
    )
    check(
        "positive gain",
        compute_gain(perfect, perfect + 10, gt, 10000) > 0,
    )
    check(
        "negative gain",
        compute_gain(perfect + 10, perfect, gt, 10000) < 0,
    )
    check(
        "identical poses give zero gain",
        compute_gain(perfect, perfect, gt, 10000) == 0,
    )

    def expect_value_error(name, callback):
        try:
            callback()
        except ValueError:
            check(name, True)
        else:
            raise AssertionError(name)

    no_labels = gt.copy()
    no_labels[:, 2] = 0
    expect_value_error(
        "reject no labeled keypoints",
        lambda: compute_instance_oks(perfect, no_labels, 10000),
    )
    expect_value_error(
        "reject zero area",
        lambda: compute_instance_oks(perfect, gt, 0),
    )
    expect_value_error(
        "reject wrong prediction shape",
        lambda: compute_instance_oks(np.zeros((16, 2)), gt, 10000),
    )
    nonfinite = perfect.copy()
    nonfinite[0, 0] = np.nan
    expect_value_error(
        "reject nonfinite predictions",
        lambda: compute_instance_oks(nonfinite, gt, 10000),
    )

    # Real COCO GT annotations, with constructed predictions.
    # These are numerical checks, not model-performance experiments.
    ann_path = args.ann
    coco = read_json(ann_path)
    split = read_json(args.split)
    manifest = read_json(args.manifest)
    if split["source_annotation"] != ann_path.name:
        raise ValueError("Annotation basename differs from split source")
    verify_manifest(split, manifest)
    check("valid-person manifest matches split", True)

    valid_dev_ids = set(manifest["dev_valid_ann_ids"])
    dev_images = set(split["dev_image_ids"])
    annotations = sorted(coco["annotations"], key=lambda ann: ann["id"])

    selected = []
    for ann in annotations:
        if ann["id"] not in valid_dev_ids:
            continue
        visibility = ann["keypoints"][2::3]
        if 1 in visibility and 2 in visibility:
            selected.append(ann)
        if len(selected) == 5:
            break

    check("five real GT samples selected", len(selected) == 5)
    rng = np.random.default_rng(42)

    for ann in selected:
        check(
            f"annotation {ann['id']} belongs to dev",
            ann["image_id"] in dev_images,
        )
        gt_real = np.asarray(ann["keypoints"], dtype=np.float64).reshape(17, 3)
        base = gt_real[:, :2].copy()
        noise_scale = 0.02 * np.sqrt(ann["area"])
        predictions = {
            "perfect": base,
            "shifted": base + 5.0,
            "noisy": base + rng.normal(0, noise_scale, base.shape),
        }
        for case, pred in predictions.items():
            compare(f"real_{ann['id']}_{case}", pred, ann)

    max_difference = max(
        row["absolute_difference"] for row in comparisons
    )
    report = {
        "purpose": "Numerical validation, not model performance evaluation",
        "reference": "pycocotools.COCOeval.computeOks",
        "absolute_tolerance": ATOL,
        "numpy_version": np.__version__,
        "split_file": args.split.name,
        "manifest_file": args.manifest.name,
        "source_annotation": ann_path.name,
        **hash_metadata(split),
        "passed_checks": passed,
        "real_annotation_ids": [ann["id"] for ann in selected],
        "official_comparisons": comparisons,
        "max_absolute_difference": max_difference,
        "limitations": (
            "Predictions are constructed from real GT. "
            "Actual P0/Pc coordinate conversion and person matching "
            "still require integration validation."
        ),
    }
    write_json(report_path, report)

    print("\nALL CHECKS PASSED")
    print("Official comparisons:", len(comparisons))
    print("Max absolute difference:", max_difference)
    print("Report:", report_path)


if __name__ == "__main__":
    main()