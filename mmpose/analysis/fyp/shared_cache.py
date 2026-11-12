"""Portable, non-pickle, identity-checked reference caches (NumPy only)."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA = "cigpose-reference-cache-v1"
SHAPES = {
    "features": (17, 256), "uncertainty": (17,), "confidence": (17,),
    "P_0_coords": (17, 2), "P_0_scores": (17,), "bbox": (4,),
    "bbox_score": (), "input_center": (2,), "input_scale": (2,),
    "input_size": (2,), "original_size": (2,), "top1_joint": (),
    "image_id": (), "annotation_id": (),
}
INTEGER_FIELDS = {"image_id", "annotation_id", "top1_joint"}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def validate_arrays(arrays):
    if set(arrays) != set(SHAPES):
        raise ValueError("Unexpected/missing cache fields; GT and gain are not cache inputs")
    n = len(arrays["annotation_id"])
    if n == 0:
        raise ValueError("Empty shard")
    for key, tail in SHAPES.items():
        a = arrays[key]
        if a.shape != (n,) + tail or not np.isfinite(a).all():
            raise ValueError(f"Invalid shape or nonfinite values: {key}")
        dtype = np.dtype("int64" if key in INTEGER_FIELDS else "float32")
        if a.dtype != dtype:
            raise ValueError(f"{key} must use {dtype}, got {a.dtype}")
    if np.any(arrays["input_scale"] <= 0) or np.any(arrays["input_size"] <= 0):
        raise ValueError("Invalid coordinate restoration scale")
    if np.any(arrays["original_size"] <= 0):
        raise ValueError("Invalid original image size")
    if np.any(arrays["bbox"][:, 2:] <= arrays["bbox"][:, :2]):
        raise ValueError("Invalid xyxy bounding box")
    if np.any((arrays["top1_joint"] < 0) | (arrays["top1_joint"] >= 17)):
        raise ValueError("Invalid joint index")
    if not np.allclose(arrays["confidence"] + arrays["uncertainty"], 1, atol=1e-6):
        raise ValueError("Confidence/uncertainty mismatch")
    if np.any((arrays["uncertainty"] < 0) | (arrays["uncertainty"] > 1)):
        raise ValueError("Invalid uncertainty")
    if not np.array_equal(arrays["uncertainty"].argmax(1), arrays["top1_joint"]):
        raise ValueError("Top-1 mismatch")
    pairs = list(zip(arrays["image_id"].tolist(), arrays["annotation_id"].tolist()))
    if len({a for _, a in pairs}) != n:
        raise ValueError("Duplicate annotation in shard")
    return pairs


class ReferenceCache:
    """Read one shard at a time. Identity is (dataset_split, image_id, annotation_id).

    Writer uses atomic shards, each tied to the complete run fingerprint. Resume
    scans committed shards, so a crash between shard and status writes loses no data.
    Only one writer may operate on a directory; .writer.lock enforces this.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.run = json.loads((self.directory / "run.json").read_text(encoding="utf-8"))
        if self.run["schema_version"] != SCHEMA:
            raise ValueError("Unsupported schema")
        self.fingerprint = digest_json(self.run)
        self.expected = {int(ann): int(img) for img, ann in self.run["expected_pairs"]}
        if len(self.expected) != len(self.run["expected_pairs"]):
            raise ValueError("Duplicate expected annotation")

    @classmethod
    def create(cls, directory, protocol, expected_pairs, resume=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pairs = sorted([list(map(int, p)) for p in expected_pairs], key=lambda p: p[1])
        if not pairs or len({p[1] for p in pairs}) != len(pairs):
            raise ValueError("Expected IDs must be nonempty and unique")
        run = {"schema_version": SCHEMA, "protocol": protocol, "expected_pairs": pairs}
        if (directory / "run.json").exists():
            if not resume:
                raise FileExistsError("Cache exists; use --resume or a new output directory")
            cache = cls(directory)
            if cache.run != run:
                raise ValueError("Resume protocol/weights/config/IDs differ from existing cache")
            return cache
        if any(directory.iterdir()):
            raise FileExistsError("New cache directory must be empty")
        atomic_json(directory / "run.json", run)
        return cls(directory)

    def shards(self):
        return sorted(self.directory.glob("shard-*.npz"))

    def load_shard(self, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != self.fingerprint:
                raise ValueError(f"Wrong run fingerprint: {path}")
            arrays = {k: data[k] for k in data.files if k != "fingerprint"}
        pairs = validate_arrays(arrays)
        for image_id, ann_id in pairs:
            if self.expected.get(ann_id) != image_id:
                raise ValueError(f"Unexpected/wrong image-person pair: {image_id}, {ann_id}")
        return arrays

    def verify(self, require_complete=False):
        seen = set()
        for path in self.shards():
            arrays = self.load_shard(path)
            ids = set(arrays["annotation_id"].tolist())
            if seen & ids:
                raise ValueError("Duplicate annotations across shards")
            seen.update(ids)
        if require_complete and seen != set(self.expected):
            raise ValueError(f"Incomplete cache: {len(seen)}/{len(self.expected)}")
        return seen

    def iter_shards(self, require_complete=True):
        self.verify(require_complete=require_complete)
        for path in self.shards():
            yield self.load_shard(path)


class CacheWriter:
    def __init__(self, cache):
        self.cache = cache

    def __enter__(self):
        self.lock = self.cache.directory / ".writer.lock"
        with self.lock.open("x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        try:
            self.seen = self.cache.verify()
        except BaseException:
            self.lock.unlink()
            raise
        return self

    def __exit__(self, *args):
        self.lock.unlink()

    def append(self, arrays):
        pairs = validate_arrays(arrays)
        ids = {a for _, a in pairs}
        if self.seen & ids:
            raise ValueError("Refusing to append previously cached instances")
        for image_id, ann_id in pairs:
            if self.cache.expected.get(ann_id) != image_id:
                raise ValueError("Wrong image/person membership")
        # Name from the first annotation, independent of dataloader batch order.
        path = self.cache.directory / f"shard-{int(arrays['annotation_id'][0]):012d}.npz"
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as f:
            np.savez(f, fingerprint=np.array(self.cache.fingerprint), **arrays)
            f.flush()
            os.fsync(f.fileno())
        # Read back exact bytes/arrays before publishing the shard.
        loaded = self.cache.load_shard(temporary)
        if any(not np.array_equal(arrays[k], loaded[k]) for k in arrays):
            raise ValueError("Cache serialization round-trip failed")
        os.replace(temporary, path)
        self.seen.update(ids)
        atomic_json(self.cache.directory / "status.json", {
            "fingerprint": self.cache.fingerprint, "count": len(self.seen),
            "expected": len(self.cache.expected),
            "complete": self.seen == set(self.cache.expected),
        })


def restore_coords(input_coords, center, scale, size):
    return input_coords / size * scale + center - 0.5 * scale


def assert_disjoint(train_cache, val_cache):
    train = ReferenceCache(train_cache)
    val = ReferenceCache(val_cache)
    if train.run["protocol"]["dataset_split"] != "train2017" or val.run["protocol"]["dataset_split"] != "val2017":
        raise ValueError("Expected train2017 and val2017 caches")
    train.verify(True)
    val.verify(True)
    if set(train.expected.values()) & set(val.expected.values()):
        raise ValueError("Train/val image leakage")
    if set(train.expected) & set(val.expected):
        raise ValueError("Train/val annotation overlap")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate cache shards without PyTorch")
    p.add_argument("cache", type=Path)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--val-cache", type=Path)
    args = p.parse_args()
    cache = ReferenceCache(args.cache)
    seen = cache.verify(require_complete=not args.allow_partial)
    if args.val_cache:
        assert_disjoint(args.cache, args.val_cache)
    print(f"PASS: {len(seen)}/{len(cache.expected)} instances; all committed shards verified")


if __name__ == "__main__":
    main()
