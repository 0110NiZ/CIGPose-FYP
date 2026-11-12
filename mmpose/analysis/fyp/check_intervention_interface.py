from pathlib import Path
import os
import sys

import torch
from mmengine.config import Config
from mmengine.runner import Runner


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
MMPOSE_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]

sys.path.insert(0, str(MMPOSE_ROOT))

CONFIG = (
    MMPOSE_ROOT
    / "projects/cigpose/body_2d_keypoint/"
      "cigpose-m_1xb384-420e_coco-256x192.py"
)

CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/cigpose-m_coco_256x192.pth"
)


def max_abs_diff(a, b):
    return float((a - b).abs().max().item())


def pair_close(pair_a, pair_b, atol=1e-7, rtol=1e-6):
    ax, ay = pair_a
    bx, by = pair_b

    return (
        torch.allclose(ax, bx, atol=atol, rtol=rtol)
        and torch.allclose(ay, by, atol=atol, rtol=rtol)
    )


def main():
    print("=== CIGPose public-interface smoke test ===")
    print("Config     :", CONFIG)
    print("Checkpoint :", CHECKPOINT)

    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    # Keep relative dataset paths consistent with existing experiments.
    os.chdir(MMPOSE_ROOT)

    cfg = Config.fromfile(str(CONFIG))

    # Runner requires an explicit work directory.
    cfg.work_dir = str(
        MMPOSE_ROOT / "work_dirs/fyp_interface_smoke_test"
    )

    # ---------------------------------------------------------
    # Fixed project protocol: single-view
    # ---------------------------------------------------------
    if "test_cfg" not in cfg.model:
        cfg.model["test_cfg"] = {}

    cfg.model["test_cfg"]["flip_test"] = False

    # Small deterministic test batch.
    cfg.test_dataloader.batch_size = 2
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    if "sampler" in cfg.test_dataloader:
        cfg.test_dataloader.sampler["shuffle"] = False

    runner = Runner.from_cfg(cfg)

    # Match the loading procedure used by the existing audit scripts.
    runner.call_hook("before_run")
    runner.load_checkpoint(str(CHECKPOINT))

    model = runner.model

    if hasattr(model, "module"):
        model = model.module

    model.eval()
    head = model.head

    print()
    print("flip_protocol: single_view")
    print("flip_test    :", cfg.model.test_cfg.flip_test)
    print("model.training:", model.training)
    print("head.training :", head.training)

    if cfg.model.test_cfg.flip_test is not False:
        raise AssertionError("flip_test must be False")

    # ---------------------------------------------------------
    # Get one real COCO batch
    # ---------------------------------------------------------
    data = next(iter(runner.test_dataloader))

    with torch.no_grad():
        processed = model.data_preprocessor(
            data,
            training=False,
        )

        feats = model.extract_feat(
            processed["inputs"]
        )

        # =====================================================
        # Test 1:
        # New state extractor vs old diagnostic implementation
        # =====================================================
        state = head.extract_refinement_state(feats)
        old_diag = head.diagnostic_forward(feats)

        features = state["features"]
        batch_size = features.shape[0]

        p0_match_old = pair_close(
            state["p0_simcc"],
            old_diag["baseline"],
        )

        uncertainty_match_old = torch.allclose(
            state["uncertainty"],
            old_diag["confound_score"],
            atol=1e-7,
            rtol=1e-6,
        )

        top1_match_old = torch.equal(
            state["top1_indices"],
            old_diag["top1_indices"],
        )

        print()
        print("TEST 1 - refinement-state compatibility")
        print("features shape        :", tuple(features.shape))
        print("uncertainty shape     :", tuple(state["uncertainty"].shape))
        print("P0 matches old baseline:", p0_match_old)
        print("uncertainty matches old:", uncertainty_match_old)
        print("Top-1 matches old      :", top1_match_old)

        # =====================================================
        # Test 2:
        # alpha = 0 must reproduce P0
        # =====================================================
        joint_ids_0 = state["top1_indices"].clone()

        alphas_0 = torch.zeros(
            batch_size,
            device=features.device,
            dtype=features.dtype,
        )

        features_before = features.clone()

        modified_0, mask_0 = head.cim.apply_intervention(
            features,
            joint_ids_0,
            alphas_0,
        )

        pc0 = head.predict_from_joint_features(
            modified_0
        )

        input_unchanged = torch.equal(
            features,
            features_before,
        )

        alpha0_features_exact = torch.equal(
            modified_0,
            features,
        )

        alpha0_p0_match = pair_close(
            pc0,
            state["p0_simcc"],
        )

        alpha0_mask_empty = not bool(
            mask_0.any().item()
        )

        print()
        print("TEST 2 - alpha = 0")
        print("input unchanged        :", input_unchanged)
        print("features exactly P0    :", alpha0_features_exact)
        print("prediction matches P0  :", alpha0_p0_match)
        print("intervention mask empty:", alpha0_mask_empty)
        print(
            "P0 x max abs diff      :",
            max_abs_diff(
                pc0[0],
                state["p0_simcc"][0],
            ),
        )
        print(
            "P0 y max abs diff      :",
            max_abs_diff(
                pc0[1],
                state["p0_simcc"][1],
            ),
        )

        # =====================================================
        # Test 3:
        # alpha = 1 must match old full replacement
        # force_intervene() uses one common joint for the batch.
        # =====================================================
        full_joint = int(
            state["top1_indices"][0].item()
        )

        joint_ids_1 = torch.full(
            (batch_size,),
            full_joint,
            device=features.device,
            dtype=torch.long,
        )

        alphas_1 = torch.ones(
            batch_size,
            device=features.device,
            dtype=features.dtype,
        )

        new_full, new_mask = head.cim.apply_intervention(
            features,
            joint_ids_1,
            alphas_1,
        )

        old_full, old_mask = head.cim.force_intervene(
            features,
            full_joint,
        )

        alpha1_features_match = torch.equal(
            new_full,
            old_full,
        )

        alpha1_mask_match = torch.equal(
            new_mask,
            old_mask,
        )

        new_full_pred = head.predict_from_joint_features(
            new_full
        )

        old_full_pred = head.predict_from_joint_features(
            old_full
        )

        alpha1_pred_match = pair_close(
            new_full_pred,
            old_full_pred,
        )

        print()
        print("TEST 3 - alpha = 1")
        print("tested joint           :", full_joint)
        print("features match old full:", alpha1_features_match)
        print("mask matches old full  :", alpha1_mask_match)
        print("prediction match       :", alpha1_pred_match)

        # =====================================================
        # Test 4:
        # Different persons use different joints and strengths
        # =====================================================
        mixed_joint_ids = (
            torch.arange(
                batch_size,
                device=features.device,
                dtype=torch.long,
            )
            % head.cim.num_keypoints
        )

        strength_choices = torch.tensor(
            [0.25, 1.00],
            device=features.device,
            dtype=features.dtype,
        )

        mixed_alphas = strength_choices[
            torch.arange(
                batch_size,
                device=features.device,
            )
            % len(strength_choices)
        ]

        mixed, mixed_mask = head.cim.apply_intervention(
            features,
            mixed_joint_ids,
            mixed_alphas,
        )

        mixed_ok = True

        for b in range(batch_size):
            joint = int(
                mixed_joint_ids[b].item()
            )
            alpha = mixed_alphas[b]

            expected = (
                (1.0 - alpha) * features[b, joint]
                + alpha
                * head.cim.canonical_features.weight[joint]
            )

            if not torch.allclose(
                mixed[b, joint],
                expected,
                atol=1e-7,
                rtol=1e-6,
            ):
                mixed_ok = False

            other = torch.ones(
                head.cim.num_keypoints,
                device=features.device,
                dtype=torch.bool,
            )
            other[joint] = False

            if not torch.equal(
                mixed[b, other],
                features[b, other],
            ):
                mixed_ok = False

            if not bool(
                mixed_mask[b, joint].item()
            ):
                mixed_ok = False

        print()
        print("TEST 4 - batch-wise independent actions")
        print("joint_ids              :", mixed_joint_ids.tolist())
        print("alphas                 :", mixed_alphas.tolist())
        print("batch-wise behaviour   :", mixed_ok)

    all_ok = all([
        p0_match_old,
        uncertainty_match_old,
        top1_match_old,
        input_unchanged,
        alpha0_features_exact,
        alpha0_p0_match,
        alpha0_mask_empty,
        alpha1_features_match,
        alpha1_mask_match,
        alpha1_pred_match,
        mixed_ok,
    ])

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
