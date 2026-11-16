"""Run Module A generator with persistent failure logs."""

import argparse
import json
import os
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def completed(output):
    path = output / "status.json"
    if not path.is_file():
        return 0

    with path.open(encoding="utf-8") as f:
        return len(json.load(f).get("done", {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    generator = (
        args.repo / "mmpose/analysis/fyp/build_a_labels.py"
    )

    if not generator.is_file():
        raise FileNotFoundError(generator)

    args.output.mkdir(parents=True, exist_ok=True)

    log_dir = args.output / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    log_path = log_dir / f"run_{run_id}.log"
    report_path = log_dir / f"run_{run_id}.json"

    command = [
        sys.executable,
        "-u",
        str(generator),
        "--repo", str(args.repo),
        "--cache", str(args.cache),
        "--checkpoint", str(args.checkpoint),
        "--output", str(args.output),
        "--max-shards", str(args.max_shards),
        "--device", args.device
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(args.repo / "mmpose")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["PYTHONUNBUFFERED"] = "1"

    before = completed(args.output)
    tail = deque(maxlen=40)

    print("=== MODULE A LOGGED RUN ===", flush=True)
    print("Completed before:", before, flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(args.repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            tail.append(line.rstrip())

        returncode = process.wait()

    after = completed(args.output)

    report = {
        "run_id": run_id,
        "success": returncode == 0,
        "return_code": returncode,
        "completed_before": before,
        "completed_after": after,
        "new_completed_shards": after - before,
        "max_shards": args.max_shards,
        "log_file": str(log_path),
        "last_output_lines": list(tail)
    }

    temp = report_path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    os.replace(temp, report_path)

    print("\n=== RUN SUMMARY ===")
    print("Return code:", returncode)
    print("Completed shards:", after)
    print("Run report:", report_path)
    print("Full log:", log_path)

    if returncode != 0:
        print("FAILED: Fix the error, then rerun to resume.")

    return returncode


if __name__ == "__main__":
    sys.exit(main())
