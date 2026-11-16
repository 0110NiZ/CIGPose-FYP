
"""Independent verification of Module A Gain labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from oks_utility import compute_instance_oks


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def audit(cache, out, expect_full):

    issues = []

    status = read_json(out / "status.json")
    cache_status = read_json(cache / "status.json")
    run = read_json(cache / "run.json")
    metadata = read_json(out / "metadata.json")
    coco = read_json(cache / "selected_annotations.json")

    annotations = {
        int(a["id"]): a
        for a in coco["annotations"]
    }

    expected = int(cache_status["expected"])

    require(
        cache_status["complete"] is True,
        "Source cache incomplete"
    )
    require(
        int(cache_status["count"]) == expected,
        "Source cache count mismatch"
    )
    require(
        len(annotations) == len(coco["annotations"]) == expected,
        "GT annotation count or identity mismatch"
    )
    require(
        metadata["source_cache_fingerprint"]
        == cache_status["fingerprint"],
        "Metadata cache fingerprint mismatch"
    )
    require(
        metadata["checkpoint_sha256"]
        == run["protocol"]["checkpoint_sha256"],
        "Metadata checkpoint mismatch"
    )

    source_names = {
        p.name for p in cache.glob("shard-*.npz")
    }

    done = status.get("done", {})

    require(
        isinstance(done, dict),
        "status.done must be a dictionary"
    )

    fields = {
        "image_id",
        "annotation_id",
        "oks_before",
        "oks_after",
        "gain"
    }

    seen = set()
    verified_shards = 0
    positive = 0
    negative = 0
    zero = 0

    declared_files = {
        record["file"] for record in done.values()
    }

    actual_files = {
        p.name for p in out.glob("shard-*.npz")
    }

    if declared_files != actual_files:
        issues.append(
            "NPZ files do not match status.done"
        )

    for source_name, record in sorted(done.items()):

        try:
            require(
                source_name in source_names,
                "Source shard missing"
            )

            source_file = cache / source_name
            label_file = out / record["file"]

            require(
                label_file.is_file(),
                "Label NPZ missing"
            )

            require(
                sha256(label_file) == record["sha256"],
                "SHA256 mismatch"
            )

            with np.load(
                source_file, allow_pickle=False
            ) as src, np.load(
                label_file, allow_pickle=False
            ) as lab:

                require(
                    set(lab.files) == fields,
                    "Incorrect five-field schema"
                )

                source_ids = src["annotation_id"]
                source_images = src["image_id"]
                p0 = src["P_0_coords"]

                ids = lab["annotation_id"]
                images = lab["image_id"]
                before = lab["oks_before"]
                after = lab["oks_after"]
                gain = lab["gain"]

                n = len(source_ids)

                require(
                    n == int(record["count"]),
                    "Person count mismatch"
                )

                require(
                    np.array_equal(ids, source_ids),
                    "Annotation IDs or order mismatch"
                )

                require(
                    np.array_equal(images, source_images),
                    "Image IDs or order mismatch"
                )

                require(
                    ids.dtype == np.int64
                    and images.dtype == np.int64,
                    "IDs must be int64"
                )

                require(
                    before.shape == (n,),
                    "oks_before shape mismatch"
                )

                require(
                    after.shape == (n, 17),
                    "oks_after shape mismatch"
                )

                require(
                    gain.shape == (n, 17),
                    "Gain shape mismatch"
                )

                require(
                    all(
                        a.dtype.kind == "f"
                        for a in (before, after, gain)
                    ),
                    "OKS and Gain must be floats"
                )

                require(
                    all(
                        np.isfinite(a).all()
                        for a in (before, after, gain)
                    ),
                    "NaN or Inf detected"
                )

                require(
                    np.all((before >= 0) & (before <= 1))
                    and np.all((after >= 0) & (after <= 1)),
                    "OKS outside [0,1]"
                )

                gain_tol = (
                    2e-6 if gain.dtype == np.float32
                    else 1e-10
                )

                require(
                    np.allclose(
                        gain,
                        after - before[:, None],
                        rtol=0,
                        atol=gain_tol
                    ),
                    "Gain formula mismatch"
                )

                # Independently recalculate OKS_before
                expected_before = []

                for i, ann_id in enumerate(source_ids):

                    ann = annotations.get(int(ann_id))

                    require(
                        ann is not None,
                        f"GT annotation missing: {ann_id}"
                    )

                    require(
                        int(ann["image_id"])
                        == int(source_images[i]),
                        "GT person/image mismatch"
                    )

                    gt = np.asarray(
                        ann["keypoints"],
                        dtype=np.float64
                    ).reshape(17, 3)

                    value = compute_instance_oks(
                        p0[i],
                        gt,
                        float(ann["area"])
                    )

                    expected_before.append(value)

                before_tol = (
                    2e-6 if before.dtype == np.float32
                    else 1e-9
                )

                require(
                    np.allclose(
                        before,
                        expected_before,
                        rtol=0,
                        atol=before_tol
                    ),
                    "oks_before mismatch"
                )

                current_ids = set(map(int, ids))

                require(
                    not (current_ids & seen),
                    "Duplicate annotation IDs across shards"
                )

                # P0 differences must be logged.
                diagnostic = (
                    out / "p0_diagnostics"
                    / f"{Path(source_name).stem}.json"
                )

                require(
                    diagnostic.is_file(),
                    "P0 diagnostics missing"
                )

                diagnostic_data = read_json(diagnostic)

                require(
                    diagnostic_data["source_shard"]
                    == source_name,
                    "P0 diagnostic shard mismatch"
                )

                require(
                    int(diagnostic_data["persons"]) == n,
                    "P0 diagnostic count mismatch"
                )

                seen.update(current_ids)
                verified_shards += 1

                positive += int(np.count_nonzero(gain > 0))
                negative += int(np.count_nonzero(gain < 0))
                zero += int(np.count_nonzero(gain == 0))

        except Exception as exc:
            issues.append(
                f"{source_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    # Full coverage must match the entire source cache.
    full_coverage = (
        len(done) == len(source_names)
        and verified_shards == len(source_names)
        and len(seen) == expected
        and seen == set(annotations)
    )

    if expect_full and not full_coverage:
        issues.append("Full coverage not achieved")

    if metadata.get("complete") is True:
        if not full_coverage:
            issues.append(
                "Metadata incorrectly claims completion"
            )

    report = {
        "passed": len(issues) == 0,
        "coverage_complete": full_coverage,
        "verified_shards": verified_shards,
        "declared_shards": len(done),
        "source_shards": len(source_names),
        "verified_persons": len(seen),
        "expected_persons": expected,
        "gain_count": positive + negative + zero,
        "positive_gains": positive,
        "negative_gains": negative,
        "zero_gains": zero,
        "issues": issues,
        "scope": (
            "Schema, IDs, hashes, coverage, Gain formula, "
            "cached P0 and independently recalculated OKS_before. "
            "Does not re-run the model for OKS_after."
        )
    }

    report_file = out / "validation_report.json"

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== MODULE A INDEPENDENT AUDIT ===")
    print("Verified shards:", verified_shards, "/", len(done))
    print("Verified persons:", len(seen), "/", expected)
    print("Gain values:", positive + negative + zero)
    print("Full coverage:", full_coverage)
    print("Issues:", len(issues))

    for issue in issues[:10]:
        print("ERROR:", issue)

    print("Report:", report_file)
    print("PASS" if not issues else "FAIL")

    return len(issues) == 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache", type=Path, required=True
    )

    parser.add_argument(
        "--output", type=Path, required=True
    )

    parser.add_argument(
        "--expect-full", action="store_true"
    )

    args = parser.parse_args()

    success = audit(
        args.cache,
        args.output,
        args.expect_full
    )

    raise SystemExit(0 if success else 1)
