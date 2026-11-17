import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT / "mmpose" / "analysis" / "fyp"
sys.path.insert(0, str(FYP))

from module_c import (
    pair_p0_and_pc,
    match_detections_to_gt_and_compute_gains,
    load_acceptance_probs,
    sweep_acceptance_thresholds,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Module C checkpoints on val2017."
    )

    parser.add_argument("--p0-file", type=Path, required=True)
    parser.add_argument("--pc-file", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)

    parser.add_argument("--prob-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--taus",
        type=str,
        default="0.4,0.5,0.6",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.start_epoch > args.end_epoch:
        raise ValueError("start-epoch must be <= end-epoch")

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading P0 / Pc once...")

    with open(args.p0_file, "r", encoding="utf-8") as f:
        p0_records = json.load(f)

    with open(args.pc_file, "r", encoding="utf-8") as f:
        pc_records = json.load(f)

    pair_p0_and_pc(p0_records, pc_records)

    print(
        "Pairing check PASSED:",
        len(p0_records),
        "detections",
    )

    print("Computing fixed validation gain audit once...")

    audit = match_detections_to_gt_and_compute_gains(
        p0_records=p0_records,
        pc_records=pc_records,
        ann_file=str(args.ann_file),
    )

    taus = [
        float(x.strip())
        for x in args.taus.split(",")
        if x.strip()
    ]

    summary = []

    for epoch in range(
        args.start_epoch,
        args.end_epoch + 1,
    ):
        print()
        print("=" * 70)
        print(f"EPOCH {epoch:03d}")
        print("=" * 70)

        prob_file = (
            args.prob_root
            / f"epoch_{epoch:03d}.npz"
        )

        if not prob_file.exists():
            raise FileNotFoundError(prob_file)

        p_acc = load_acceptance_probs(
            str(prob_file),
            p0_records,
        )

        results = sweep_acceptance_thresholds(
            p0_records=p0_records,
            pc_records=pc_records,
            p_acc=p_acc,
            taus=taus,
            ann_file=str(args.ann_file),
            gains=audit["gains"],
            matched_mask=audit["matched_mask"],
        )

        best = max(
            results,
            key=lambda r: r["metrics"]["coco/AP"],
        )

        row = {
            "epoch": epoch,
            "best_tau": best["tau"],
            "best_ap": best["metrics"]["coco/AP"],
            "best_ar": best["metrics"]["coco/AR"],
            "decision_analysis": best["decision_analysis"],
        }

        summary.append(row)

        with open(
            args.output / f"epoch_{epoch:03d}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "epoch": epoch,
                    "threshold_sweep": results,
                },
                f,
                indent=2,
            )

        print()
        print(
            f"EPOCH {epoch:03d} BEST: "
            f"tau={row['best_tau']:.2f} "
            f"AP={row['best_ap']:.12f} "
            f"AR={row['best_ar']:.12f}"
        )

    summary.sort(
        key=lambda r: r["best_ap"],
        reverse=True,
    )

    with open(
        args.output / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    top1_ap = 0.7354021675321071
    p0_ap = 0.7362229928153036

    print()
    print("=" * 78)
    print("FINAL AP RANKING")
    print("=" * 78)

    print(
        f"{'rank':>4} "
        f"{'epoch':>5} "
        f"{'tau':>6} "
        f"{'AP':>14} "
        f"{'APx100':>10} "
        f"{'vsTop1':>10} "
        f"{'vsP0':>10}"
    )

    for rank, row in enumerate(summary, 1):
        ap = row["best_ap"]

        print(
            f"{rank:4d} "
            f"{row['epoch']:5d} "
            f"{row['best_tau']:6.2f} "
            f"{ap:14.12f} "
            f"{ap * 100:10.4f} "
            f"{(ap - top1_ap) * 100:+10.4f} "
            f"{(ap - p0_ap) * 100:+10.4f}"
        )

    best = summary[0]

    print()
    print("===== OVERALL BEST =====")
    print("epoch:", best["epoch"])
    print("tau:", best["best_tau"])
    print("AP:", best["best_ap"])
    print("AP x100:", best["best_ap"] * 100)
    print("AR:", best["best_ar"])
    print(
        "vs Top-1:",
        (best["best_ap"] - top1_ap) * 100,
    )
    print(
        "vs P0:",
        (best["best_ap"] - p0_ap) * 100,
    )


if __name__ == "__main__":
    main()
