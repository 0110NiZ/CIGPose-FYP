"""Prepare Module C training data."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import config

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
sys.path.insert(0, str(FYP))

from module_c import encode_module_c_features
from shared_cache import ReferenceCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=config.TRAIN_CACHE,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=config.LABELS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.TRAIN_DATA,
    )
    args = parser.parse_args()

    cache = ReferenceCache(args.train_cache)
    labels = np.load(args.labels)

    if len(cache.expected) != 149813:
        raise ValueError(
            f"Expected 149813 cached people, got {len(cache.expected)}."
        )

    if len(labels["annotation_id"]) != 149813:
        raise ValueError("Training label count is not 149813.")

    label_index = {
        int(ann_id): i
        for i, ann_id in enumerate(labels["annotation_id"])
    }

    all_x = []
    all_y = []
    all_gain = []
    all_ids = []

    count = 0

    for shard_path in cache.shards():
        arrays = cache.load_shard(shard_path)
        ann_ids = arrays["annotation_id"].astype(np.int64)

        try:
            pos = np.asarray(
                [label_index[int(x)] for x in ann_ids],
                dtype=np.int64,
            )
        except KeyError as e:
            raise ValueError(
                f"Missing label for annotation {e.args[0]}"
            ) from e

        joint_ids = labels["joint_id"][pos].astype(np.int64)

        if not np.array_equal(
            joint_ids,
            arrays["top1_joint"].astype(np.int64),
        ):
            raise ValueError(
                "Joint IDs do not match the train cache."
            )

        x = encode_module_c_features(
            p0_coords=torch.as_tensor(
                arrays["P_0_coords"],
                dtype=torch.float32,
            ),
            pc_coords=torch.as_tensor(
                labels["pc_coords"][pos],
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
                labels["pc_scores"][pos],
                dtype=torch.float32,
            ),
            joint_ids=torch.as_tensor(
                joint_ids,
                dtype=torch.long,
            ),
            alphas=torch.as_tensor(
                labels["alpha"][pos],
                dtype=torch.float32,
            ),
            uncertainty=torch.as_tensor(
                arrays["uncertainty"],
                dtype=torch.float32,
            ),
        )

        all_x.append(x.numpy().astype(np.float32))
        all_y.append(labels["label"][pos].astype(np.float32))
        all_gain.append(labels["gain"][pos].astype(np.float64))
        all_ids.append(ann_ids)

        count += len(ann_ids)

        if count % 8192 < len(ann_ids):
            print(f"Prepared {count} people", flush=True)

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    gain = np.concatenate(all_gain)
    annotation_id = np.concatenate(all_ids)

    if x.shape != (149813, 137):
        raise ValueError(
            f"Unexpected Module C input shape: {x.shape}"
        )

    if y.shape != (149813,):
        raise ValueError("Unexpected target shape.")

    if len(np.unique(annotation_id)) != 149813:
        raise ValueError("Duplicate annotation IDs.")

    if not np.isfinite(x).all():
        raise ValueError("Non-finite Module C inputs.")

    if not np.isfinite(gain).all():
        raise ValueError("Non-finite gain values.")

    if not np.array_equal(
        y.astype(np.int64),
        (gain > 0.0).astype(np.int64),
    ):
        raise ValueError("Targets do not match gain labels.")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        args.output,
        x=x,
        y=y,
        gain=gain,
        annotation_id=annotation_id,
    )

    print()
    print("Training people:", len(y))
    print("Input shape:", x.shape)
    print("Positive:", int(y.sum()))
    print("Negative or zero:", int(len(y) - y.sum()))
    print("Positive ratio:", float(y.mean()))
    print("Saved to:", args.output)
    print("MODULE C TRAIN DATA: PASS")


if __name__ == "__main__":
    main()
