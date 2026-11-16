
"""Finalize Module A metadata only after full independent audit."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def check_committed(repo, path):
    relative = path.relative_to(repo).as_posix()

    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files",
         "--error-unmatch", "--", relative],
        capture_output=True
    )

    if tracked.returncode != 0:
        raise RuntimeError("File not committed: " + relative)

    changed = subprocess.run(
        ["git", "-C", str(repo), "diff",
         "--quiet", "HEAD", "--", relative]
    )

    if changed.returncode != 0:
        raise RuntimeError("Uncommitted changes: " + relative)


def finalize(repo, cache, out):
    meta_path = out / "metadata.json"

    meta = read_json(meta_path)
    status = read_json(out / "status.json")
    cache_status = read_json(cache / "status.json")

    if meta["complete"] is True:
        raise RuntimeError("Metadata is already complete")

    source_names = {
        p.name for p in cache.glob("shard-*.npz")
    }

    done = status.get("done", {})
    expected = int(meta["expected_count"])

    count = sum(
        int(record["count"])
        for record in done.values()
    )

    print("=== FULL COVERAGE PRECHECK ===")
    print("Source shards:", len(source_names))
    print("Completed shards:", len(done))
    print("Completed persons:", count, "/", expected)

    if (
        cache_status.get("complete") is not True
        or set(done) != source_names
        or count != expected
        or expected != int(cache_status["count"])
        or expected != int(cache_status["expected"])
    ):
        print("SAFE STOP: Full coverage not achieved")
        print("Metadata remains incomplete")
        return 2

    folder = repo / "mmpose/analysis/fyp"
    generator = folder / "build_a_labels.py"
    checker = folder / "check_a_labels.py"
    oks = folder / "oks_utility.py"

    if sha256(generator) != status["generator_sha256"]:
        raise RuntimeError("Generator SHA256 mismatch")

    if sha256(oks) != status["oks_sha256"]:
        raise RuntimeError("OKS SHA256 mismatch")

    # Metadata must identify a real committed generator.
    check_committed(repo, generator)
    check_committed(repo, checker)

    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True
    ).strip()

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(repo / "mmpose")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    print("\n=== INDEPENDENT FULL AUDIT ===")

    audit = subprocess.run(
        [
            sys.executable, str(checker),
            "--cache", str(cache),
            "--output", str(out),
            "--expect-full"
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True
    )

    print(audit.stdout)

    if audit.returncode != 0:
        print(audit.stderr[-2000:])
        print("SAFE STOP: Independent audit failed")
        return 3

    report = read_json(out / "validation_report.json")

    if not (
        report.get("passed") is True
        and report.get("coverage_complete") is True
        and report.get("verified_persons") == expected
        and report.get("verified_shards") == len(source_names)
        and report.get("gain_count") == expected * 17
        and not report.get("issues")
    ):
        print("SAFE STOP: Invalid full-audit report")
        return 4

    # Exact shard structure required by N's contract.
    manifest = sorted(
        [
            {
                "file": record["file"],
                "sha256": record["sha256"],
                "count": int(record["count"])
            }
            for record in done.values()
        ],
        key=lambda item: item["file"]
    )

    if len({item["file"] for item in manifest}) != len(manifest):
        raise RuntimeError("Duplicate output filenames")

    # Change metadata only after every check passes.
    meta["count"] = expected
    meta["generator_git_commit"] = commit
    meta["oks_implementation_sha256"] = sha256(oks)
    meta["shards"] = manifest
    meta["complete"] = True

    temporary = meta_path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    os.replace(temporary, meta_path)

    print("\nPASS: Metadata finalized")
    print("Persons:", expected)
    print("Shards:", len(manifest))
    print("Generator commit:", commit)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    try:
        result = finalize(
            args.repo.resolve(),
            args.cache.resolve(),
            args.output.resolve()
        )
        sys.exit(result)
    except Exception as exc:
        print("SAFE STOP:", type(exc).__name__, str(exc))
        sys.exit(1)
