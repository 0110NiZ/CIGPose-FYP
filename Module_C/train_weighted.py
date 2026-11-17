import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
sys.path.insert(0, str(FYP))

import config

from module_c import ModuleCAcceptanceClassifier
from precision_policy import configure_legacy_precision


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=1e-3,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    configure_legacy_precision(seed=config.SEED)

    args.output.mkdir(parents=True, exist_ok=True)

    data = np.load(config.TRAIN_DATA)

    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    gain = data["gain"].astype(np.float32)

    # Larger gain magnitude gets more training weight.
    weight = np.minimum(
        np.abs(gain) / args.gain_scale,
        1.0,
    ).astype(np.float32)

    # Keep the overall loss scale similar to ordinary BCE.
    positive_weights = weight[weight > 0]

    if len(positive_weights) == 0:
        raise RuntimeError("All training weights are zero.")

    mean_weight = float(weight.mean())

    if mean_weight <= 0:
        raise RuntimeError("Invalid mean training weight.")

    weight = weight / mean_weight

    print("Training samples:", len(x))
    print("Input dimension:", x.shape[1])
    print("Positive labels:", int(y.sum()))
    print("Negative/zero labels:", int((y == 0).sum()))
    print("Gain scale:", args.gain_scale)
    print("Raw mean weight:", mean_weight)
    print(
        "Zero-weight samples:",
        int((weight == 0).sum()),
    )

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(weight),
    )

    generator = torch.Generator()
    generator.manual_seed(config.SEED)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    device = torch.device(config.DEVICE)

    model = ModuleCAcceptanceClassifier(
        in_dim=x.shape[1],
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="none"
    )

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        loss_sum = 0.0
        correct = 0
        seen = 0

        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)

            per_sample_loss = criterion(
                logits,
                batch_y,
            )

            loss = (
                per_sample_loss * batch_w
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = batch_x.size(0)

            loss_sum += (
                float(loss.detach().cpu())
                * batch_size
            )

            pred = (
                torch.sigmoid(logits) > 0.5
            ).long()

            correct += int(
                (pred == batch_y.long())
                .sum()
                .item()
            )

            seen += batch_size

        row = {
            "epoch": epoch,
            "loss": loss_sum / seen,
            "accuracy": correct / seen,
            "samples": seen,
        }

        history.append(row)
        print(json.dumps(row))

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "input_dim": x.shape[1],
            "hidden_dim": 128,
            "dropout": 0.1,
            "gain_scale": args.gain_scale,
            "training": "gain_weighted_bce",
        }

        torch.save(
            checkpoint,
            args.output / f"epoch_{epoch:03d}.pt",
        )

        torch.save(
            checkpoint,
            args.output / "last.pt",
        )

        with open(
            args.output / "history.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                history,
                f,
                indent=2,
            )

    print("MODULE C WEIGHTED TRAINING: PASS")


if __name__ == "__main__":
    main()
