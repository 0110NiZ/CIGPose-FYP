"""Train the Module C acceptance classifier."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
sys.path.insert(0, str(FYP))

from module_c import ModuleCAcceptanceClassifier
from precision_policy import configure_legacy_precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=config.TRAIN_DATA,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUT,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.WEIGHT_DECAY,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only the first N samples for a training smoke test.",
    )
    parser.add_argument(
        "--device",
        default=config.DEVICE,
    )
    args = parser.parse_args()

    configure_legacy_precision(seed=config.SEED)

    d = np.load(args.data)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.float32)

    if x.shape != (149813, 137):
        raise ValueError(f"Unexpected input shape: {x.shape}")

    if y.shape != (149813,):
        raise ValueError(f"Unexpected target shape: {y.shape}")

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive.")
        x = x[:args.max_samples]
        y = y[:args.max_samples]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    dataset = TensorDataset(x, y)

    generator = torch.Generator()
    generator.manual_seed(config.SEED)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = ModuleCAcceptanceClassifier(
        in_dim=137,
        hidden_dim=128,
        dropout=0.1,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    args.output.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total = 0
        loss_sum = 0.0
        correct = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += int((pred == batch_y).sum())

            n = len(batch_y)
            total += n
            loss_sum += float(loss.detach()) * n

        row = {
            "epoch": epoch,
            "loss": loss_sum / total,
            "accuracy": correct / total,
            "samples": total,
        }
        history.append(row)

        print(json.dumps(row), flush=True)

        checkpoint = {
            "epoch": epoch,
            "input_dim": 137,
            "hidden_dim": 128,
            "dropout": 0.1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "seed": config.SEED,
            "history": history,
        }

        torch.save(
            checkpoint,
            args.output / "last.pt",
        )

        # Save every epoch so val2017 can select the best checkpoint.
        torch.save(
            checkpoint,
            args.output / f"epoch_{epoch:03d}.pt",
        )

    (args.output / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("MODULE C TRAINING: PASS")


if __name__ == "__main__":
    main()
