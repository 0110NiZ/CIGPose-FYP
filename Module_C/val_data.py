"""Prepare Module C inputs for val2017 detections."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import config

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
sys.path.insert(0, str(FYP))

from detection_cache import ReferenceCache
from module_c import encode_module_c_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-cache",
        type=Path,
        default=config.VAL_CACHE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.VAL_DATA,
    )
    args = parser.parse_args()

    cache = ReferenceCache(args.val_cache)

    if len(cache.expected) != 104125:
        raise ValueError(
            f"Expected 104125 detections, got {len(cache.expected)}."
        )

    all_x = []
    all_image_ids = []
    all_detection_ids = []
    all_joint_ids = []
    count = 0

    for shard_path in cache.shards():
        arrays = cache.load_shard(shard_path)

        n = len(arrays["source_detection_index"])

        x = encode_module_c_features(
            p0_coords=torch.as_tensor(
                arrays["P_0_coords"],
                dtype=torch.float32,
            ),
            pc_coords=torch.as_tensor(
                arrays["top1_coords"],
                dtype=torch.float32,
            ),
            bboxes_xyxy=torch.as_tensor(
                arrays["bbox"],
                dtype=torch.float32,
            ),
            p0_scores=torch.as_tensor(
                arrays["P_0_scores"],
                dtype=torch.float32,
            ),
            pc_scores=torch.as_tensor(
                arrays["top1_scores"],
                dtype=torch.float32,
            ),
            joint_ids=torch.as_tensor(
                arrays["top1_joint"],
                dtype=torch.long,
            ),
            alphas=torch.ones(
                n,
                dtype=torch.float32,
            ),
            uncertainty=torch.as_tensor(
                arrays["uncertainty"],
                dtype=torch.float32,
            ),
        )

        all_x.append(x.numpy().astype(np.float32))
        all_image_ids.append(
            arrays["image_id"].astype(np.int64)
        )
        all_detection_ids.append(
            arrays["source_detection_index"].astype(np.int64)
        )
        all_joint_ids.append(
            arrays["top1_joint"].astype(np.int64)
        )

        count += n

        if count % 8192 < n:
            print(f"Prepared {count} detections", flush=True)

    x = np.concatenate(all_x)
    image_id = np.concatenate(all_image_ids)
    detection_id = np.concatenate(all_detection_ids)
    joint_id = np.concatenate(all_joint_ids)

    if x.shape != (104125, 137):
        raise ValueError(
            f"Unexpected Module C val shape: {x.shape}"
        )

    if len(np.unique(detection_id)) != 104125:
        raise ValueError("Duplicate detection IDs.")

    if detection_id.min() != 0 or detection_id.max() != 104124:
        raise ValueError("Unexpected detection index range.")

    if not np.isfinite(x).all():
        raise ValueError("Non-finite validation inputs.")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        args.output,
        x=x,
        image_id=image_id,
        source_detection_index=detection_id,
        joint_id=joint_id,
    )

    print()
    print("Validation detections:", len(x))
    print("Input shape:", x.shape)
    print(
        "Detection index range:",
        int(detection_id.min()),
        "to",
        int(detection_id.max()),
    )
    print("Saved to:", args.output)
    print("MODULE C VAL DATA: PASS")


if __name__ == "__main__":
    main()
