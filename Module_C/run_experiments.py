"""
Module C Experiment Runner (Multi-seed, Multi-task, Soft-target, Summary Features)
Compatible with CPU & GPU.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
sys.path.insert(0, str(FYP))

from module_c import ModuleCAcceptanceClassifier  # noqa: E402
from precision_policy import configure_legacy_precision  # noqa: E402


def append_summary_features(x: np.ndarray) -> np.ndarray:
    """Append 8 selected-joint & whole-pose summary features to the 137-D input."""
    delta_norm = x[:, 34:68].reshape(-1, 17, 2)
    p0_scores = x[:, 68:85]
    delta_scores = x[:, 85:102]
    joint_onehot = x[:, 102:119]
    uncertainty = x[:, 120:137]

    disp_norm = np.linalg.norm(delta_norm, axis=-1)  # [N, 17]

    sel_disp = np.sum(disp_norm * joint_onehot, axis=-1, keepdims=True)
    sel_p0_score = np.sum(p0_scores * joint_onehot, axis=-1, keepdims=True)
    sel_delta_score = np.sum(delta_scores * joint_onehot, axis=-1, keepdims=True)
    sel_unc = np.sum(uncertainty * joint_onehot, axis=-1, keepdims=True)

    mean_disp = np.mean(disp_norm, axis=-1, keepdims=True)
    max_disp = np.max(disp_norm, axis=-1, keepdims=True)
    mean_p0_score = np.mean(p0_scores, axis=-1, keepdims=True)
    mean_delta_score = np.mean(delta_scores, axis=-1, keepdims=True)

    return np.concatenate(
        [
            x,
            sel_disp,
            sel_p0_score,
            sel_delta_score,
            sel_unc,
            mean_disp,
            max_disp,
            mean_p0_score,
            mean_delta_score,
        ],
        axis=-1,
    ).astype(np.float32)


class MultiTaskModuleC(nn.Module):
    """Shared MLP trunk with binary acceptance head + auxiliary gain regression head."""

    def __init__(self, in_dim: int = 137, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.cls_head = nn.Linear(hidden_dim // 2, 1)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.cls_head(h).squeeze(-1)
        pred_gain = self.reg_head(h).squeeze(-1)
        return logits, pred_gain


@torch.no_grad()
def predict_val_probs(model: nn.Module, val_x: np.ndarray, device: torch.device, is_multitask: bool) -> np.ndarray:
    model.eval()
    probs = []
    bsz = 2048
    for i in range(0, len(val_x), bsz):
        bx = torch.from_numpy(val_x[i:i + bsz]).to(device)
        out = model(bx)
        logits = out[0] if is_multitask else out
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--val-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", type=str, default="weighted_bce", choices=["weighted_bce", "multitask", "soft_target"])
    p.add_argument("--add-summary-features", action="store_true", help="Append 8 summary features (137->145D)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gain-scale", type=float, default=1e-3)
    p.add_argument("--reg-weight", type=float, default=0.5)
    p.add_argument("--soft-temp", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    configure_legacy_precision(seed=args.seed)

    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    ckpt_dir = args.output / "checkpoints"
    prob_dir = args.output / "probs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    train_npz = np.load(args.train_data, allow_pickle=False)
    x = train_npz["x"].astype(np.float32)
    y = train_npz["y"].astype(np.float32)
    gain = train_npz["gain"].astype(np.float32)

    val_npz = np.load(args.val_data, allow_pickle=False)
    val_x = val_npz["x"].astype(np.float32)
    val_ids = val_npz["detection_instance_id"] if "detection_instance_id" in val_npz else None

    if args.add_summary_features:
        x = append_summary_features(x)
        val_x = append_summary_features(val_x)

    print(f"Running mode={args.mode} | seed={args.seed} | in_dim={x.shape[1]} | device={device}")

    weight = np.minimum(np.abs(gain) / args.gain_scale, 1.0).astype(np.float32)
    weight = weight / float(weight.mean())

    scaled_gain = np.clip(gain * 100.0, -5.0, 5.0).astype(np.float32)
    y_soft = (1.0 / (1.0 + np.exp(-np.clip(gain / args.soft_temp, -20.0, 20.0)))).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(y_soft),
        torch.from_numpy(weight),
        torch.from_numpy(scaled_gain),
    )
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=gen)

    is_multitask = args.mode == "multitask"
    model = (
        MultiTaskModuleC(in_dim=x.shape[1]) if is_multitask
        else ModuleCAcceptanceClassifier(in_dim=x.shape[1], hidden_dim=128, dropout=0.1)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="mean")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for bx, by, by_soft, bw, bg in loader:
            bx, by, by_soft, bw, bg = bx.to(device), by.to(device), by_soft.to(device), bw.to(device), bg.to(device)
            optimizer.zero_grad(set_to_none=True)

            if is_multitask:
                logits, pred_g = model(bx)
                cls_loss = (bce_none(logits, by) * bw).mean()
                reg_loss = huber(pred_g, bg)
                loss = cls_loss + args.reg_weight * reg_loss
            elif args.mode == "soft_target":
                logits = model(bx)
                loss = bce_none(logits, by_soft).mean()
            else:
                logits = model(bx)
                loss = (bce_none(logits, by) * bw).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bsz = bx.size(0)
            loss_sum += float(loss.detach().cpu()) * bsz
            correct += int(((torch.sigmoid(logits) > 0.5).long() == by.long()).sum().item())
            seen += bsz

        val_probs = predict_val_probs(model, val_x, device, is_multitask)
        save_dict = {"p_acc": val_probs}
        if val_ids is not None:
            save_dict["detection_instance_id"] = val_ids
        np.savez(prob_dir / f"epoch_{epoch:03d}.npz", **save_dict)

        torch.save({"epoch": epoch, "model_state": model.state_dict(), "mode": args.mode, "seed": args.seed}, ckpt_dir / f"epoch_{epoch:03d}.pt")
        row = {"epoch": epoch, "loss": round(loss_sum / seen, 6), "train_acc": round(correct / seen, 5)}
        history.append(row)
        print(json.dumps(row))

    with open(args.output / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"DONE! Val probs saved to: {prob_dir}")


if __name__ == "__main__":
    main()