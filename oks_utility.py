"""Instance-level COCO-17 OKS for offline gain supervision.

Inputs:
    pred_kpts: [17, 2], original-image coordinates.
    gt_kpts:   [17, 3], original-image x, y and COCO visibility.
    area:      positive area from the same GT annotation.

No prediction confidence threshold is applied.
GT points with v=1 and v=2 both participate.
Instances without labeled keypoints must be filtered by the caller.
This utility does not replace standard COCO AP/AR evaluation.
"""

import numpy as np


# COCO joint order: nose, eyes, ears, shoulders, elbows,
# wrists, hips, knees, ankles; left precedes right in each pair.
COCO_SIGMAS = np.array(
    [
        0.26, 0.25, 0.25, 0.35, 0.35,
        0.79, 0.79, 0.72, 0.72, 0.62, 0.62,
        1.07, 1.07, 0.87, 0.87, 0.89, 0.89,
    ],
    dtype=np.float64,
) / 10.0
COCO_SIGMAS.setflags(write=False)


def compute_instance_oks(pred_kpts, gt_kpts, area):
    """Return one person's OKS as a Python float.

    Raises ValueError for invalid inputs instead of creating
    a misleading zero-OKS or zero-gain training label.

    For PyTorch tensors, the caller should first convert with:
        tensor.detach().cpu().numpy()
    """
    pred = np.asarray(pred_kpts, dtype=np.float64)
    gt = np.asarray(gt_kpts, dtype=np.float64)

    if pred.shape != (17, 2):
        raise ValueError(
            f"pred_kpts must have shape (17, 2), got {pred.shape}"
        )
    if gt.shape != (17, 3):
        raise ValueError(
            f"gt_kpts must have shape (17, 3), got {gt.shape}"
        )

    area_array = np.asarray(area, dtype=np.float64)
    if area_array.ndim != 0:
        raise ValueError("area must be a scalar")
    area_value = float(area_array)
    if not np.isfinite(area_value) or area_value <= 0:
        raise ValueError("GT area must be finite and positive")

    visibility = gt[:, 2]
    if not np.all(np.isin(visibility, [0, 1, 2])):
        raise ValueError("COCO visibility must be 0, 1 or 2")

    labeled = visibility > 0
    if not labeled.any():
        raise ValueError("No labeled keypoints: filter this instance")

    if not np.isfinite(pred).all():
        raise ValueError("Predicted coordinates must be finite")
    if not np.isfinite(gt[labeled, :2]).all():
        raise ValueError("Labeled GT coordinates must be finite")

    # Same per-keypoint expression as COCOeval.computeOks
    # for a fixed GT with at least one labeled keypoint.
    dx = pred[labeled, 0] - gt[labeled, 0]
    dy = pred[labeled, 1] - gt[labeled, 1]
    variances = (COCO_SIGMAS[labeled] * 2.0) ** 2
    e = (
        (dx ** 2 + dy ** 2)
        / variances
        / (area_value + np.spacing(1))
        / 2.0
    )
    return float(np.exp(-e).mean())


def compute_gain(P_c, P_0, gt_kpts, area):
    """Return OKS(P_c, GT) - OKS(P_0, GT).

    P_c and P_0 must belong to the same person and use
    the same original-image coordinate system.
    """
    oks_before = compute_instance_oks(P_0, gt_kpts, area)
    oks_after = compute_instance_oks(P_c, gt_kpts, area)
    return oks_after - oks_before