#!/usr/bin/env python3
"""One-command entry point for Module A ground-truth Gain label generation.

Place this file at the root of the CIGPose-FYP-ModuleA repository.

Requires the existing, tested implementations at:
    mmpose/analysis/fyp/build_a_labels.py
    mmpose/analysis/fyp/check_a_labels.py
    mmpose/analysis/fyp/run_a_labels_logged.py
    mmpose/analysis/fyp/finalize_a_labels.py

The entry point *orchestrates* those implementations; it intentionally does
not reproduce CIGPose inference, P0 reconstruction, interventions or OKS.

Usage (inside the configured CIGPose Python environment):
    python main.py --cache /path/to/train2017 \
        --checkpoint /path/to/cigpose-m.pth \
        --output /path/to/train_a_labels

For an incomplete smoke test:
    python main.py --cache ... --checkpoint ... --output /path/to/smoke \
        --max-shards 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Basic text progress fallback if tqdm is unavailable.


SCRIPTS = {
    "runner": "run_a_labels_logged.py",
    "generator": "build_a_labels.py",
    "checker": "check_a_labels.py",
    "finalizer": "finalize_a_labels.py",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def run_stage(name: str, command: list[str], repo: Path, env: dict[str, str]) -> int:
    """Run audit/finalize stages; generation uses run_generation instead."""
    print(f"\n{'=' * 18} {name} {'=' * 18}", flush=True)
    try:
        result = subprocess.run(command, cwd=repo, env=env, check=False)
        print(f"{name} exit code: {result.returncode}", flush=True)
        return result.returncode
    except OSError as exc:
        print(f"ERROR: Failed to start {name}: {exc}", file=sys.stderr, flush=True)
        return 1


def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def milestones(total: int) -> list[int]:
    """Sequential, non-overlapping windows: 1, 10, 50, 150, 250, ..."""
    targets = [n for n in (1, 10, 50, 150) if n <= total]
    if total > 150:
        targets.extend(range(250, total + 1, 100))
    if not targets or targets[-1] != total:
        targets.append(total)
    return sorted(set(targets))


def shard_status(output: Path) -> dict:
    """Atomic status writes mean a read sees either old or new JSON."""
    path = output / "status.json"
    if not path.exists():
        return {"done": {}, "failed": {}}
    try:
        obj = read_json(path)
        if isinstance(obj, dict):
            return obj
    except (OSError, ValueError):
        pass
    return {"done": {}, "failed": {}}


def progress_summary(output: Path, source: Path, index: int, total: int,
                     state: dict) -> str:
    """Metrics from saved labels/diagnostics, not a second model pass."""
    label = f"[{index}/{total}]"
    record = state.get("done", {}).get(source.name)
    if record is None:
        error = state.get("failed", {}).get(source.name, {}).get("error_message")
        if error is not None:
            return f"{label} FAILED: {error}"
        return f"{label} Finished window; last shard not completed"

    persons = int(record["count"])
    diagnostic = output / "p0_diagnostics" / f"{source.stem}.json"
    try:
        err = float(read_json(diagnostic)["p0_max_abs_error_px"])
        err_text = f"{err:.5f}px"
    except (OSError, ValueError, TypeError, KeyError):
        err_text = "unavailable"

    try:
        import numpy as np
        with np.load(output / record["file"], allow_pickle=False) as arrays:
            mean_gain = float(np.mean(arrays["gain"]))
        gain_text = f"{mean_gain:+.6f}"
    except (OSError, ValueError, TypeError, KeyError, ImportError):
        gain_text = "unavailable"

    return (f"{label} {persons} persons | P0 error={err_text} "
            f"| Mean gain={gain_text}")


def stop_process_tree(process: subprocess.Popen) -> None:
    """Stop runner AND its nested generator when the notebook is cancelled."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def run_generation(command: list[str], repo: Path, env: dict[str, str],
                   output: Path, sources: list[Path]) -> int:
    """Quietly stream generator logs to disk, display one window bar at a time.

    Status is updated by build_a_labels.py after every successful shard.
    Failed attempts are also reflected in status.json. This monitor never
    changes labels or the generator's resume decisions.
    """
    total = len(sources)
    status_start = shard_status(output)
    initial_done = set(status_start.get("done", {}))
    initial_attempts = {
        key: int(value.get("attempts", 0))
        for key, value in status_start.get("failed", {}).items()
    }

    print("\n=== GENERATE + RECORD LOGS ===", flush=True)
    print(f"Resume: {len(initial_done)}/{total} shards already Done", flush=True)
    print("One progress bar per group; completed bars are cleared.", flush=True)

    log_path = output / "main_generation_console.log"
    output.mkdir(parents=True, exist_ok=True)
    pending_messages: list[str] = []
    messages_lock = threading.Lock()
    reader_error: list[Exception] = []

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== GENERATION RUN {now_utc()} ===\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
            start_new_session=(os.name == "posix"),
        )

        def pump() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    stripped = line.strip()
                    if stripped.startswith(("[FAILED", "FATAL:", "ERROR:", "Traceback")):
                        with messages_lock:
                            pending_messages.append(stripped)
            except Exception as exc:
                reader_error.append(exc)

        worker = threading.Thread(target=pump, daemon=True)
        worker.start()

        boundaries = milestones(total)
        bound_index = 0
        begin = 1
        current = None
        last_shown = 0
        last_fallback = 0.0

        def clear_bar() -> None:
            nonlocal current
            if current is not None:
                current.close()
                current = None

        def open_bar(start: int, end: int, already: int):
            if tqdm is None:
                return None
            return tqdm(
                total=end - start + 1, initial=already,
                desc=f"Shards {start}-{end}",
                unit="shard", leave=False,
                dynamic_ncols=True, mininterval=0.5,
                file=sys.stdout,
            )

        def flush_messages():
            with messages_lock:
                items = pending_messages[:]
                pending_messages.clear()
            if not items:
                return
            clear_bar()
            for item in items:
                print(item, flush=True)

        prior_sigterm = signal.getsignal(signal.SIGTERM)

        def on_sigterm(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, on_sigterm)
        try:
            while True:
                state = shard_status(output)
                done = set(state.get("done", {}))
                failed = state.get("failed", {})
                attempted_failed = {
                    key for key, value in failed.items()
                    if int(value.get("attempts", 0))
                    > initial_attempts.get(key, 0)
                }
                attempted = done | attempted_failed

                # Advance only one window at a time; remove its bar before
                # creating the next one (not four simultaneous bars).
                while bound_index < len(boundaries):
                    end = boundaries[bound_index]
                    window_sources = sources[begin - 1:end]
                    completed_count = sum(p.name in attempted for p in window_sources)
                    window_size = len(window_sources)

                    # Resume: don't replay fully completed older windows.
                    old_window = all(p.name in initial_done for p in window_sources)
                    if old_window:
                        clear_bar()
                        begin = end + 1
                        bound_index += 1
                        last_shown = 0
                        continue

                    if (current is None and completed_count < window_size
                            and not (begin == 1 and end == 1)):
                        current = open_bar(begin, end, completed_count)
                        last_shown = completed_count
                    elif current is not None and completed_count > last_shown:
                        current.update(completed_count - last_shown)
                        last_shown = completed_count

                    if current is None and tqdm is None:
                        t = time.monotonic()
                        if t - last_fallback > 3:
                            print(f"\rShards {begin}-{end}: "
                                  f"{completed_count}/{window_size}",
                                  end="", flush=True)
                            last_fallback = t

                    if completed_count == window_size:
                        clear_bar()
                        if tqdm is None:
                            print()
                        print(progress_summary(output, sources[end - 1], end,
                                               total, state), flush=True)
                        begin = end + 1
                        bound_index += 1
                        last_shown = 0
                        continue
                    break

                flush_messages()
                if process.poll() is not None:
                    break
                time.sleep(0.7)

            process.wait()
            worker.join()
            # Final iteration picks up the latest saved status.
            state = shard_status(output)
            done = set(state.get("done", {}))
            failed = state.get("failed", {})
            print(f"\nGenerate exit code: {process.returncode}", flush=True)
            print(f"Saved shards: {len(done)}/{total} | "
                  f"Failed: {len(failed)}", flush=True)
            print("Full generation log:", log_path, flush=True)
            if reader_error:
                print("WARNING: Log reader error:", reader_error[0], flush=True)
                return 1
            return int(process.returncode)

        except KeyboardInterrupt:
            stop_process_tree(process)
            worker.join()
            print("\nInterrupted. Saved shards remain available for Resume.",
                  flush=True)
            return 130
        finally:
            clear_bar()
            signal.signal(signal.SIGTERM, prior_sigterm)



def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate, verify, and (when complete) finalize Module A labels."
    )
    parser.add_argument(
        "--config", type=Path,
        help="Optional JSON config. If no arguments are supplied, use "
             "module_a_run_config.json beside main.py automatically.",
    )
    parser.add_argument(
        "--repo", type=Path,
        help="CIGPose-FYP-ModuleA repository root (default: directory of main.py).",
    )
    parser.add_argument(
        "--cache", type=Path,
        help="Extracted Train2017 GT-box shared cache directory.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Frozen CIGPose-M checkpoint used to construct the shared cache.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="New or resumable label output directory.",
    )
    parser.add_argument(
        "--device", help="Torch device for candidate inference (default: cuda:0)."
    )
    parser.add_argument(
        "--max-shards", type=int,
        help="Only process the first N source shards; 0 means all shards (default).",
    )
    parser.add_argument(
        "--no-finalize", action="store_true",
        help="Run full independent audit without finalizing metadata or requiring Git commit.",
    )
    args = parser.parse_args()
    wall_start = time.monotonic()

    config_path = (args.config.expanduser().resolve() if args.config else
                   Path(__file__).resolve().parent / "module_a_run_config.json")
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = read_json(config_path)
            if not isinstance(config, dict):
                parser.error("Configuration JSON must be an object")
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot load configuration {config_path}: {exc}")
    elif args.config:
        parser.error(f"Config file not found: {config_path}")

    def setting(key: str) -> Any:
        from_cli = getattr(args, key)
        return from_cli if from_cli is not None else config.get(key)

    def config_path_value(key: str, default: Path | None = None) -> Path:
        raw = setting(key)
        if raw is None:
            if default is None:
                parser.error(
                    f"Missing --{key.replace('_', '-')} (or '{key}' in "
                    "module_a_run_config.json)"
                )
            return default
        value = Path(raw).expanduser()
        # Relative paths in a JSON config are relative to that JSON file.
        # Relative CLI paths are relative to the current working directory.
        if not value.is_absolute() and getattr(args, key) is None:
            value = config_path.parent / value
        return value.resolve()

    repo = config_path_value("repo", Path(__file__).resolve().parent)
    cache = config_path_value("cache")
    checkpoint = config_path_value("checkpoint")
    output = config_path_value("output")
    device = setting("device") or "cuda:0"
    max_shards = setting("max_shards")
    max_shards = int(max_shards) if max_shards is not None else 0
    scripts_dir = repo / "mmpose" / "analysis" / "fyp"

    if max_shards < 0:
        parser.error("--max-shards must be >= 0")
    if not repo.is_dir():
        parser.error(f"Repository not found: {repo}")
    if not cache.is_dir():
        parser.error(f"Cache directory not found: {cache}")
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    missing = [filename for filename in SCRIPTS.values()
               if not (scripts_dir / filename).is_file()]
    if missing:
        parser.error("Missing Module A scripts in " + str(scripts_dir)
                     + ": " + ", ".join(missing))

    source_shards = sorted(cache.glob("shard-*.npz"))
    if not source_shards:
        parser.error(f"No source shard-*.npz files found in {cache}")

    # Use an explicit positive limit. This does not assume that the underlying
    # generator and runner interpret --max-shards 0 in any particular way.
    shard_limit = min(max_shards or len(source_shards), len(source_shards))
    is_full_run = shard_limit == len(source_shards)

    metadata_file = output / "metadata.json"
    if metadata_file.is_file():
        try:
            old_meta = read_json(metadata_file)
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read existing output metadata: {exc}")
        if old_meta.get("complete") is True:
            print("This output directory is already marked complete.", file=sys.stderr)
            print("Do not regenerate over finalized labels; use a new --output.", file=sys.stderr)
            return 2

    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "mmpose") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"

    summary_path = output / "main_pipeline_summary.json"
    summary: dict[str, Any] = {
        "started_utc": now_utc(),
        "finished_utc": None,
        "complete": False,
        "repo": str(repo),
        "cache": str(cache),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "device": device,
        "source_shards": len(source_shards),
        "selected_shards": shard_limit,
        "full_run": is_full_run,
        "stages": {},
        "stage_seconds": {},
    }

    def finish(code: int) -> int:
        summary["finished_utc"] = now_utc()
        elapsed = time.monotonic() - wall_start
        summary["elapsed_seconds"] = round(elapsed, 3)
        summary["elapsed_hms"] = human_duration(elapsed)
        summary["exit_code"] = code
        status_file = output / "status.json"
        if status_file.is_file():
            try:
                status = read_json(status_file)
                summary["done_shards"] = len(status.get("done", {}))
                summary["failed_shards"] = len(status.get("failed", {}))
                summary["done_persons"] = sum(
                    int(row["count"]) for row in status.get("done", {}).values()
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                summary["status_read_error"] = str(exc)
        if metadata_file.is_file():
            try:
                meta = read_json(metadata_file)
                summary["complete"] = meta.get("complete") is True
                summary["metadata_count"] = meta.get("count")
            except (OSError, ValueError) as exc:
                summary["metadata_read_error"] = str(exc)
        # Append completed attempts, so a later Resume can report the sum of
        # recorded sessions. Forced termination cannot be timed reliably.
        history_path = output / "main_run_history.jsonl"
        cumulative = 0.0
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    cumulative += float(json.loads(line).get("elapsed_seconds", 0))
                except (ValueError, TypeError):
                    pass
        cumulative += elapsed
        summary["cumulative_recorded_seconds"] = round(cumulative, 3)
        summary["cumulative_recorded_hms"] = human_duration(cumulative)
        atomic_json(summary_path, summary)
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps({
                "started_utc": summary["started_utc"],
                "finished_utc": summary["finished_utc"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "exit_code": code,
                "done_shards": summary.get("done_shards"),
                "failed_shards": summary.get("failed_shards"),
            }, ensure_ascii=False) + "\n")
        print("\n=== TIME SUMMARY ===", flush=True)
        for name, seconds in summary["stage_seconds"].items():
            print(f"{name}: {human_duration(seconds)}", flush=True)
        print(f"This run: {human_duration(elapsed)}", flush=True)
        print(f"Recorded runs total: {human_duration(cumulative)}", flush=True)
        print("(Forced-stop time from older runs may be absent.)", flush=True)
        print("\nPipeline summary:", summary_path, flush=True)
        print("Label directory:", output, flush=True)
        return code

    print("=== MODULE A LABEL PIPELINE ===")
    print("Total source shards:", len(source_shards))
    print("Selected source shards:", shard_limit)
    print("Mode:", "FULL" if is_full_run else "SMOKE / PARTIAL")
    print("Using Python:", sys.executable)
    print("Frozen checkpoint:", checkpoint)
    if config:
        print("Config:", config_path)
    print("Output:", output)

    # The existing generator handles cached P0, rebuilt P0 diagnostics,
    # independent alpha=1 Pj candidates, true OKS gains, status, retries,
    # and failure_history.jsonl. The wrapper captures full run logs.
    generation_cmd = [
        sys.executable, "-u", str(scripts_dir / SCRIPTS["runner"]),
        "--repo", str(repo),
        "--cache", str(cache),
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--max-shards", str(shard_limit),
        "--device", device,
    ]
    stage_start = time.monotonic()
    code = run_generation(generation_cmd, repo, env, output,
                          source_shards[:shard_limit])
    summary["stage_seconds"]["generate"] = round(time.monotonic() - stage_start, 3)
    summary["stages"]["generate"] = code
    if code != 0:
        print("Generation not fully successful. See output/status.json and run_logs/.")
        print("Fix failed shards and rerun this same command to resume.")
        print("No completeness claim or finalization attempted.")
        return finish(code)

    if not is_full_run:
        # A smoke test is useful, but must never set metadata.complete=true.
        check_cmd = [
            sys.executable, "-u", str(scripts_dir / SCRIPTS["checker"]),
            "--cache", str(cache), "--output", str(output),
        ]
        stage_start = time.monotonic()
        code = run_stage("SMOKE AUDIT", check_cmd, repo, env)
        summary["stage_seconds"]["smoke_audit"] = round(time.monotonic() - stage_start, 3)
        summary["stages"]["smoke_audit"] = code
        print("Partial run: finalization deliberately disabled.")
        return finish(code)

    if args.no_finalize:
        check_cmd = [
            sys.executable, "-u", str(scripts_dir / SCRIPTS["checker"]),
            "--cache", str(cache), "--output", str(output), "--expect-full",
        ]
        stage_start = time.monotonic()
        code = run_stage("FULL INDEPENDENT AUDIT (NO FINALIZE)",
                         check_cmd, repo, env)
        summary["stage_seconds"]["full_audit"] = round(
            time.monotonic() - stage_start, 3)
        summary["stages"]["full_audit"] = code
        print("Metadata intentionally left incomplete: no Git commit required.")
        return finish(code)

    # The existing finalizer invokes check_a_labels.py --expect-full itself,
    # verifies total coverage and hashes, then writes finalized metadata.
    finalize_cmd = [
        sys.executable, "-u", str(scripts_dir / SCRIPTS["finalizer"]),
        "--repo", str(repo),
        "--cache", str(cache),
        "--output", str(output),
    ]
    stage_start = time.monotonic()
    code = run_stage("FULL AUDIT + FINALIZE", finalize_cmd, repo, env)
    summary["stage_seconds"]["full_audit_finalize"] = round(time.monotonic() - stage_start, 3)
    summary["stages"]["full_audit_finalize"] = code
    if code != 0:
        print("Full audit/finalization failed. No completion claim made.")
        print("Review validation_report.json and the finalizer's errors.")
        return finish(code)

    try:
        meta = read_json(metadata_file)
        if meta.get("complete") is not True:
            print("ERROR: Finalizer exited 0 without complete=true.", file=sys.stderr)
            return finish(4)
    except (OSError, ValueError) as exc:
        print("ERROR: Finalized metadata cannot be read:", exc, file=sys.stderr)
        return finish(4)

    print("\nPASS: Full Module A label pipeline completed and finalized.")
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
