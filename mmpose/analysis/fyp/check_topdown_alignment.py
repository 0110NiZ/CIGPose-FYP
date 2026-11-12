from copy import deepcopy
from pathlib import Path
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner


THIS_FILE = Path(__file__).resolve()
MMPOSE_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]

CONFIG = (
    MMPOSE_ROOT
    / "projects/cigpose/body_2d_keypoint/"
      "cigpose-m_1xb384-420e_coco-256x192.py"
)

CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/cigpose-m_coco_256x192.pth"
)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def close(a, b, atol=1e-6):
    return np.allclose(
        to_numpy(a),
        to_numpy(b),
        atol=atol,
        rtol=1e-6,
    )


def main():
    print("=== Top-down alignment smoke test ===")

    os.chdir(MMPOSE_ROOT)

    cfg = Config.fromfile(str(CONFIG))

    cfg.work_dir = str(
        MMPOSE_ROOT / "work_dirs/fyp_topdown_alignment_test"
    )

    # Fixed FYP protocol
    cfg.model.test_cfg.flip_test = False

    cfg.test_dataloader.batch_size = 2
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_dataloader.sampler.shuffle = False

    runner = Runner.from_cfg(cfg)

    runner.call_hook("before_run")
    runner.load_checkpoint(str(CHECKPOINT))

    model = runner.model
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    head = model.head

    dataloader = runner.test_dataloader
    dataset = dataloader.dataset

    data = next(iter(dataloader))

    with torch.no_grad():
        processed = model.data_preprocessor(
            data,
            training=False,
        )

        feats = model.extract_feat(
            processed["inputs"]
        )

        state = head.extract_refinement_state(feats)

        # Decode P0 from SimCC into model-input coordinates.
        decoded = head.decode(
            state["p0_simcc"]
        )

    data_samples = processed["data_samples"]

    # Save values before TopdownPoseEstimator modifies them.
    input_keypoints = [
        to_numpy(pred.keypoints).copy()
        for pred in decoded
    ]

    input_scores = [
        to_numpy(pred.keypoint_scores).copy()
        for pred in decoded
    ]

    samples_for_output = deepcopy(data_samples)

    results = model.add_pred_to_datasample(
        decoded,
        None,
        samples_for_output,
    )

    all_ok = True

    print()
    print("flip_protocol: single_view")
    print("flip_test    :", cfg.model.test_cfg.flip_test)

    for b, result in enumerate(results):
        meta = result.metainfo

        center = np.asarray(
            meta["input_center"],
            dtype=np.float32,
        )

        scale = np.asarray(
            meta["input_scale"],
            dtype=np.float32,
        )

        input_size = np.asarray(
            meta["input_size"],
            dtype=np.float32,
        )

        expected_original = (
            input_keypoints[b][..., :2]
            / input_size
            * scale
            + center
            - 0.5 * scale
        )

        actual_original = to_numpy(
            result.pred_instances.keypoints[..., :2]
        )

        coords_ok = np.allclose(
            expected_original,
            actual_original,
            atol=1e-5,
            rtol=1e-6,
        )

        scores_ok = np.array_equal(
            input_scores[b],
            to_numpy(
                result.pred_instances.keypoint_scores
            ),
        )

        bbox_ok = close(
            result.pred_instances.bboxes,
            result.gt_instances.bboxes,
        )

        bbox_score_ok = close(
            result.pred_instances.bbox_scores,
            result.gt_instances.bbox_scores,
        )

        # The sampler is deterministic and shuffle=False,
        # so the first batch corresponds to dataset indices 0, 1.
        info = dataset.get_data_info(b)

        sample_img_id = meta.get(
            "img_id",
            meta.get("image_id", None),
        )

        dataset_img_id = info.get(
            "img_id",
            info.get("image_id", None),
        )

        if (
            sample_img_id is not None
            and dataset_img_id is not None
        ):
            image_id_ok = (
                sample_img_id == dataset_img_id
            )
        else:
            image_id_ok = True

        ann_id = info.get(
            "id",
            info.get(
                "ann_id",
                info.get("annotation_id", None),
            ),
        )

        print()
        print(f"PERSON {b}")
        print("dataset annotation id :", ann_id)
        print("dataset image id      :", dataset_img_id)
        print("sample image id       :", sample_img_id)
        print("coordinate restore    :", coords_ok)
        print("keypoint scores       :", scores_ok)
        print("bbox alignment        :", bbox_ok)
        print("bbox score alignment  :", bbox_score_ok)
        print("image identity        :", image_id_ok)
        print(
            "max coordinate diff   :",
            float(
                np.max(
                    np.abs(
                        expected_original
                        - actual_original
                    )
                )
            ),
        )

        person_ok = all([
            coords_ok,
            scores_ok,
            bbox_ok,
            bbox_score_ok,
            image_id_ok,
        ])

        all_ok = all_ok and person_ok

    print()
    print("=" * 55)
    print(
        "OVERALL:",
        "PASS" if all_ok else "FAIL",
    )
    print("=" * 55)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
