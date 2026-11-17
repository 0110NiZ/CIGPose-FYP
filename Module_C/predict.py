"""Run Module C on the fixed val2017 inputs."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import config

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(FYP))
sys.path.insert(0, str(SCRIPTS))

from detection_cache import ReferenceCache
from module_c import ModuleCAcceptanceClassifier
from precision_policy import configure_legacy_precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=config.VAL_DATA)
    parser.add_argument("--val-cache", type=Path, default=config.VAL_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    configure_legacy_precision(seed=config.SEED)

    d = np.load(args.data)

    x = d["x"].astype(np.float32)
    detection_index = d["source_detection_index"].astype(np.int64)

    if x.shape != (104125, 137):
        raise ValueError(f"Unexpected validation shape: {x.shape}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    model = ModuleCAcceptanceClassifier(
        in_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(args.device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probs = []

    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            end = min(start + args.batch_size, len(x))

            batch = torch.from_numpy(
                x[start:end]
            ).to(args.device)

            p = model.predict_proba(batch)

            probs.append(
                p.cpu().numpy().astype(np.float32)
            )

    p_acc = np.concatenate(probs)

    if p_acc.shape != (104125,):
        raise ValueError(
            f"Unexpected probability shape: {p_acc.shape}"
        )

    if not np.isfinite(p_acc).all():
        raise ValueError("Non-finite probabilities.")

    if np.any((p_acc < 0.0) | (p_acc > 1.0)):
        raise ValueError("Probabilities outside [0, 1].")

    cache = ReferenceCache(args.val_cache)

    if len(cache.expected) != 104125:
        raise ValueError("Unexpected validation cache size.")

    detection_instance_id = np.asarray(
        [
            cache.instance_id(int(idx))
            for idx in detection_index
        ]
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        args.output,
        p_acc=p_acc,
        detection_instance_id=detection_instance_id,
        source_detection_index=detection_index,
    )

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Detections:", len(p_acc))
    print("Probability min:", float(p_acc.min()))
    print("Probability max:", float(p_acc.max()))
    print("Probability mean:", float(p_acc.mean()))
    print("Saved to:", args.output)
    print("MODULE C PREDICTION: PASS")


if __name__ == "__main__":
    main()
