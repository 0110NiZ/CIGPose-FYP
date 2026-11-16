"""Generate Module C training labels from the train2017 GT cache."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

import config

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
MMPOSE = ROOT / "mmpose"

sys.path.insert(0, str(FYP))
sys.path.insert(0, str(MMPOSE))

from build_shared_cache import candidate_from_cache
from oks_utility import compute_gain
from precision_policy import configure_legacy_precision
from shared_cache import ReferenceCache


def load_base_model(checkpoint, device):
    """Load the frozen CIGPose model used to generate candidates."""
    from mmengine.config import Config
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules

    register_all_modules()

    cfg_path = (
        MMPOSE
        / "projects/cigpose/body_2d_keypoint/"
        "cigpose-m_1xb384-420e_coco-256x192.py"
    )

    cfg = Config.fromfile(str(cfg_path))
    cfg.model.setdefault("test_cfg", {})["flip_test"] = False

    model = init_model(cfg, str(checkpoint), device=device)
    model.requires_grad_(False)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, default=config.TRAIN_CACHE)
    parser.add_argument("--ann", type=Path, default=config.TRAIN_ANN)
    parser.add_argument("--checkpoint", type=Path, default=config.BASE_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.WORK / "labels" / "train2017_c_labels.npz",
    )
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit the number of people for a smoke test.",
    )
    args = parser.parse_args()

    configure_legacy_precision(seed=config.SEED)

    cache = ReferenceCache(args.train_cache)

    # We have already validated this cache separately.
    if len(cache.expected) != 149813:
        raise ValueError(
            f"Expected 149813 train persons, got {len(cache.expected)}."
        )

    coco = COCO(str(args.ann))

    # Build the lookup once instead of calling loadAnns for every person.
    ann_by_id = {
        int(ann["id"]): ann
        for ann in coco.dataset["annotations"]
    }

    model = load_base_model(args.checkpoint, args.device)

    annotation_ids = []
    joint_ids_all = []
    pc_coords_all = []
    pc_scores_all = []
    gains = []
    labels = []

    total = 0

    with torch.inference_mode():
        for shard_path in cache.shards():
            arrays = cache.load_shard(shard_path)

            if args.max_instances is not None:
                remaining = args.max_instances - total
                if remaining <= 0:
                    break

                if len(arrays["annotation_id"]) > remaining:
                    arrays = {
                        key: value[:remaining]
                        for key, value in arrays.items()
                    }

            joint_ids = torch.as_tensor(
                arrays["top1_joint"],
                dtype=torch.long,
                device=args.device,
            )

            alphas = torch.ones(
                len(joint_ids),
                dtype=torch.float32,
                device=args.device,
            )

            pc_coords, pc_scores = candidate_from_cache(
                model.head,
                arrays,
                joint_ids,
                alphas,
                args.device,
            )

            for i, ann_id in enumerate(arrays["annotation_id"]):
                ann_id = int(ann_id)

                if ann_id not in ann_by_id:
                    raise KeyError(f"Annotation {ann_id} not found.")

                ann = ann_by_id[ann_id]

                gt = np.asarray(
                    ann["keypoints"],
                    dtype=np.float64,
                ).reshape(17, 3)

                gain = compute_gain(
                    np.asarray(pc_coords[i], dtype=np.float64),
                    np.asarray(
                        arrays["P_0_coords"][i],
                        dtype=np.float64,
                    ),
                    gt,
                    float(ann["area"]),
                )

                annotation_ids.append(ann_id)
                joint_ids_all.append(
                    int(arrays["top1_joint"][i])
                )
                pc_coords_all.append(pc_coords[i])
                pc_scores_all.append(pc_scores[i])
                gains.append(float(gain))
                labels.append(1 if gain > 0.0 else 0)

            total += len(arrays["annotation_id"])

            if total % 1024 < len(arrays["annotation_id"]):
                print(
                    f"Generated {total} labels",
                    flush=True,
                )

    if total == 0:
        raise RuntimeError("No training labels were generated.")

    annotation_ids = np.asarray(
        annotation_ids,
        dtype=np.int64,
    )
    joint_ids_all = np.asarray(
        joint_ids_all,
        dtype=np.int64,
    )
    pc_coords_all = np.asarray(
        pc_coords_all,
        dtype=np.float32,
    )
    pc_scores_all = np.asarray(
        pc_scores_all,
        dtype=np.float32,
    )
    gains = np.asarray(
        gains,
        dtype=np.float64,
    )
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if len(annotation_ids) != total:
        raise ValueError("Label count mismatch.")

    if len(np.unique(annotation_ids)) != total:
        raise ValueError("Duplicate annotation IDs found.")

    if not np.isfinite(pc_coords_all).all():
        raise ValueError("Non-finite candidate coordinates found.")

    if not np.isfinite(pc_scores_all).all():
        raise ValueError("Non-finite candidate scores found.")

    if not np.isfinite(gains).all():
        raise ValueError("Non-finite gain values found.")

    if not np.array_equal(
        labels,
        (gains > 0.0).astype(np.int64),
    ):
        raise ValueError("Labels do not match gain > 0.")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        annotation_id=annotation_ids,
        joint_id=joint_ids_all,
        alpha=np.ones(total, dtype=np.float32),
        pc_coords=pc_coords_all,
        pc_scores=pc_scores_all,
        gain=gains,
        label=labels,
    )

    positive = int(labels.sum())
    negative_or_zero = int(total - positive)

    summary = {
        "count": total,
        "positive": positive,
        "negative_or_zero": negative_or_zero,
        "positive_ratio": positive / total,
        "mean_gain": float(gains.mean()),
        "output": str(args.output),
    }

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
