"""Migrate legacy raw-file checksums without rerunning or changing the split.

Existing statistics and OKS results are retained, not presented as new runs.
"""
import argparse
import hashlib
from pathlib import Path
from fyp_data_common import (SPLIT_HASH_METHOD, check_outputs, hash_metadata,
                             read_json, write_json)

ROOT = Path(__file__).resolve().parents[1]


def migrate(split_path, manifest_path, data_report_path, oks_report_path, out_dir, overwrite=False):
    source_bytes = split_path.read_bytes()
    split = read_json(split_path)
    manifest = read_json(manifest_path)
    data_report = read_json(data_report_path)
    oks_report = read_json(oks_report_path)
    # Accept verified LF/CRLF forms of the supplied file, not arbitrary mismatches.
    text = source_bytes.decode("utf-8-sig").replace("\r\n", "\n")
    forms = [source_bytes, text.encode("utf-8"), text.replace("\n", "\r\n").encode("utf-8")]
    legacy_hashes = {hashlib.sha256(b).hexdigest() for b in forms}
    metadata = hash_metadata(split)
    values = [manifest, data_report, oks_report]
    for value in values:
        method = value.get("split_hash_method")
        if method == SPLIT_HASH_METHOD:
            if value["split_sha256"] != metadata["split_sha256"]:
                raise ValueError("Canonical checksum mismatch; migration refused")
        elif method is None:
            if value["split_sha256"] not in legacy_hashes:
                raise ValueError("Legacy checksum cannot be verified against the split; migration refused")
        else:
            raise ValueError("Unknown checksum method; migration refused")
    outputs = [out_dir / "splits/train_dev_valid_ann_ids_v2.json",
               out_dir / "splits/train_dev_data_check_v2.json",
               out_dir / "experiments/oks/oks_validation.json"]
    if split_path.resolve() in {p.resolve() for p in outputs}:
        raise ValueError("Migration must not write to split")
    check_outputs(outputs, overwrite)
    for value, output in zip(values, outputs):
        if value.get("split_hash_method") != SPLIT_HASH_METHOD:
            value["checksum_metadata_migration"] = {
                "previous_method": "sha256-raw-file-bytes",
                "previous_split_sha256": value["split_sha256"],
                "results_recomputed": False,
                "note": "Only checksum metadata migrated; previous results retained.",
            }
        value.update(metadata)
        write_json(output, value)
    if split_path.read_bytes() != source_bytes:
        raise RuntimeError("Split unexpectedly changed")
    return metadata, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "splits/train_dev_split.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/train_dev_valid_ann_ids_v2.json")
    parser.add_argument("--data-report", type=Path, default=ROOT / "splits/train_dev_data_check_v2.json")
    parser.add_argument("--oks-report", type=Path, default=ROOT / "experiments/oks/oks_validation.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output root, separate from inputs by default")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata, outputs = migrate(args.split, args.manifest, args.data_report,
                                args.oks_report, args.out_dir, args.overwrite)
    print(metadata)
    for p in outputs:
        print("Saved:", p)
    print("PASS: checksum metadata migrated; split and previous numeric results unchanged.")


if __name__ == "__main__":
    main()
