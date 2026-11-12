"""Build GT-box single-view reference caches using T's public head interface.

Run in the existing CIGPose environment. No model training, gain labels or GT
keypoints are written into the model-input cache. See docs/fyp/shared_cache.md.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from shared_cache import (ReferenceCache, CacheWriter, SHAPES, INTEGER_FIELDS,
                          digest_json, hash_file, restore_coords)

ROOT = Path(__file__).resolve().parents[3]
MMPOSE = ROOT / "mmpose"
EXPECTED_CHECKPOINT = "3dd69c8ab4e3c7a923a0183fd9abf844cb7ef13d39d26693baa082ca7931b3d0"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def select_annotations(coco, manifest, split, annotation_name):
    """Reject legacy internal splits; require an explicit full-split ID manifest.

    L owns filtering. This function validates membership and basic validity,
    but never silently invents a new filter or fills missing IDs.
    """
    if annotation_name != f"person_keypoints_{split}.json":
        raise ValueError("Annotation filename does not match official dataset split")
    if manifest.get("dataset_split") != split or manifest.get("source_annotation") != annotation_name:
        raise ValueError("Need L's full-split manifest with dataset_split/source_annotation; not old 90/10")
    if manifest.get("filter_version") != "supervision_filter_v2":
        raise ValueError("Expected filter_version=supervision_filter_v2")
    ids = manifest.get("valid_ann_ids")
    if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("valid_ann_ids must contain unique integer annotation IDs")
    images = {im["id"]: im for im in coco["images"]}
    anns = {ann["id"]: ann for ann in coco["annotations"]}
    if len(images) != len(coco["images"]) or len(anns) != len(coco["annotations"]):
        raise ValueError("Duplicate IDs in source annotations")
    selected = []
    for ann_id in sorted(ids):
        if ann_id not in anns:
            raise ValueError(f"Missing annotation {ann_id}")
        a = anns[ann_id]
        if a["image_id"] not in images or a.get("category_id") != 1 or a.get("iscrowd", 0):
            raise ValueError(f"Invalid person/image membership: {ann_id}")
        k = np.asarray(a.get("keypoints", []), dtype=float)
        if k.shape != (51,) or not np.isfinite(k).all():
            raise ValueError(f"Invalid keypoints: {ann_id}")
        v = k[2::3]
        if not np.isin(v, [0, 1, 2]).all() or not (v > 0).any() or a.get("num_keypoints", 0) <= 0:
            raise ValueError(f"Invalid visibility: {ann_id}")
        if not np.isfinite(a.get("area", 0)) or a.get("area", 0) <= 0:
            raise ValueError(f"Invalid area: {ann_id}")
        box = np.asarray(a.get("bbox", []), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError(f"Invalid bbox: {ann_id}")
        selected.append(a)
    return selected, images


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def decode_original(model, simcc, samples):
    # Use the actual configured codec and TopdownPoseEstimator restoration.
    return model.add_pred_to_datasample(model.head.decode(simcc), None, deepcopy(samples))


def pack_batch(state, results, expected, images):
    rows = {k: [] for k in SHAPES}
    # Transfer each tensor once, not once per person in the batch.
    cpu_state = {k: array(state[k]) for k in
                 ("features", "uncertainty", "confidence", "top1_indices")}
    for b, result in enumerate(results):
        meta, pred = result.metainfo, result.pred_instances
        if "id" not in meta or "img_id" not in meta:
            raise ValueError("Missing annotation/image ID; cannot infer identity from loader position")
        ann_id, img_id = int(meta["id"]), int(meta["img_id"])
        if expected.get(ann_id) != img_id:
            raise ValueError("Model sample does not match approved image/person manifest")
        image = images[img_id]
        values = {
            "image_id": img_id, "annotation_id": ann_id,
            "features": cpu_state["features"][b],
            "uncertainty": cpu_state["uncertainty"][b],
            "confidence": cpu_state["confidence"][b],
            "top1_joint": cpu_state["top1_indices"][b],
            "P_0_coords": array(pred.keypoints)[0, :, :2],
            "P_0_scores": array(pred.keypoint_scores)[0],
            "bbox": array(pred.bboxes)[0],
            "bbox_score": array(pred.bbox_scores).reshape(-1)[0],
            "input_center": meta["input_center"], "input_scale": meta["input_scale"],
            "input_size": meta["input_size"],
            "original_size": [image["width"], image["height"]],
        }
        for key in rows:
            rows[key].append(values[key])
    return {k: np.asarray(v, dtype=np.int64 if k in INTEGER_FIELDS else np.float32)
            for k, v in rows.items()}


def candidate_from_cache(head, arrays, joint_ids, alphas, device):
    """Reloaded features -> independent candidate -> same codec/original coords."""
    import torch
    features = torch.as_tensor(arrays["features"], device=device)
    modified, _ = head.cim.apply_intervention(features, joint_ids, alphas)
    decoded = head.decode(head.predict_from_joint_features(modified))
    coords, scores = [], []
    for b, pred in enumerate(decoded):
        coords.append(restore_coords(array(pred.keypoints)[0, :, :2],
                                    arrays["input_center"][b], arrays["input_scale"][b],
                                    arrays["input_size"][b]))
        scores.append(array(pred.keypoint_scores)[0])
    return np.asarray(coords), np.asarray(scores)


def check_reloaded(model, arrays, samples, state, device):
    """Use reloaded reference features for P0 and a mixed-action candidate."""
    import torch
    n = len(samples)
    joints = torch.arange(n, device=device) % 17
    zeros = torch.zeros(n, device=device)
    coords, scores = candidate_from_cache(model.head, arrays, joints, zeros, device)
    np.testing.assert_allclose(coords, arrays["P_0_coords"], atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(scores, arrays["P_0_scores"], atol=1e-6, rtol=1e-6)
    alphas = torch.tensor([0.25, 0.5, 0.75, 1.0], device=device)[torch.arange(n, device=device) % 4]
    modified, _ = model.head.cim.apply_intervention(state["features"], joints, alphas)
    live = decode_original(model, model.head.predict_from_joint_features(modified), samples)
    coords, scores = candidate_from_cache(model.head, arrays, joints, alphas, device)
    np.testing.assert_allclose(coords, np.stack([array(r.pred_instances.keypoints)[0, :, :2] for r in live]), atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(scores, np.stack([array(r.pred_instances.keypoint_scores)[0] for r in live]), atol=1e-6, rtol=1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train2017", "val2017"], required=True)
    p.add_argument("--ann", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--valid-persons", type=Path, help="Default: manifests/<split>_valid_persons.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=MMPOSE / "projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/cigpose-m_coco_256x192.pth")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-instances", type=int, help="Smoke cache only; use a separate output directory")
    args = p.parse_args()
    if args.valid_persons is None:
        args.valid_persons = ROOT / "manifests" / f"{args.split}_valid_persons.json"
    if args.batch_size <= 0 or args.workers < 0 or (args.max_instances is not None and args.max_instances <= 0):
        p.error("Invalid batch size/workers/max-instances")
    for key in ["ann", "images", "valid_persons", "output", "config", "checkpoint"]:
        setattr(args, key, getattr(args, key).resolve())
    coco, manifest = read_json(args.ann), read_json(args.valid_persons)
    selected, images = select_annotations(coco, manifest, args.split, args.ann.name)
    annotation_hash = hash_file(args.ann)
    if manifest.get("source_annotation_sha256") != annotation_hash:
        raise ValueError("Manifest must record the SHA-256 of this exact annotation file")
    full_count = len(selected)
    if args.max_instances:
        selected = selected[:args.max_instances]
    checkpoint_hash = hash_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise ValueError("Checkpoint differs from the agreed CIGPose-m protocol")

    sys.path.insert(0, str(MMPOSE))
    os.chdir(MMPOSE)
    import torch
    from torch.utils.data import DataLoader, Subset
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmpose.apis import init_model
    from mmpose.registry import DATASETS
    from mmpose.utils import register_all_modules
    register_all_modules()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = Config.fromfile(str(args.config))
    cfg.model.setdefault("test_cfg", {})["flip_test"] = False
    if cfg.model.test_cfg.get("output_keypoint_indices") is not None:
        raise ValueError("Cache requires all 17 keypoints")
    if tuple(cfg.codec.input_size) != (192, 256):
        raise ValueError("Expected input_size=(192,256), width x height")
    pipeline = deepcopy(cfg.val_pipeline)
    if [x["type"] for x in pipeline] != ["LoadImage", "GetBBoxCenterScale", "TopdownAffine", "PackPoseInputs"]:
        raise ValueError("Only the reviewed deterministic validation pipeline is supported")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        import hashlib
        tracked_diff_hash = hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD", "--", "mmpose", "scripts"], cwd=ROOT)).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        tracked_diff_hash = "unavailable"
    # Hash implementation, including local uncommitted cache code.
    code_files = [Path(__file__), Path(__file__).with_name("shared_cache.py"),
                  MMPOSE / "mmpose/models/heads/coord_cls_heads/cig_head.py",
                  MMPOSE / "mmpose/models/pose_estimators/topdown.py"]
    protocol = {
        "dataset_split": args.split, "data_protocol": "full_train2017_val2017_v1",
        "bbox_source": "gt", "bbox_format": "xyxy", "coordinate_system": "original_image",
        "flip_protocol": "single_view", "flip_test": False, "base_model_eval": True,
        "precision": "float32", "seed": 42, "git_commit": commit,
        "tracked_code_diff_sha256": tracked_diff_hash,
        "config_sha256": digest_json(cfg.to_dict()), "checkpoint_sha256": checkpoint_hash,
        "filter_version": manifest["filter_version"],
        "annotation_sha256": annotation_hash, "valid_persons_sha256": digest_json(manifest),
        "implementation_sha256": {str(f.relative_to(ROOT)).replace('\\', '/'): hash_file(f) for f in code_files},
        "pipeline": pipeline, "batch_size": args.batch_size, "device": args.device,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "scope": "smoke" if args.max_instances else "full", "full_valid_person_count": full_count,
    }
    expected = {a["id"]: a["image_id"] for a in selected}
    cache = ReferenceCache.create(args.output, protocol, [(i, a) for a, i in expected.items()], args.resume)
    with CacheWriter(cache) as writer:
        remaining = set(expected) - writer.seen
        if not remaining:
            cache.verify(True)
            print("PASS: cache already complete")
            return
        filtered_path = args.output / "selected_annotations.json"
        filtered = {k: v for k, v in coco.items() if k not in {"images", "annotations"}}
        filtered.update(images=[images[i] for i in sorted(set(expected.values()))], annotations=selected)
        from shared_cache import atomic_json
        atomic_json(filtered_path, filtered)
        dataset = DATASETS.build(dict(type="CocoDataset", data_root="", ann_file=str(filtered_path),
                                     data_prefix=dict(img=str(args.images) + os.sep), data_mode="topdown",
                                     test_mode=True, pipeline=pipeline, serialize_data=False))
        dataset.full_init()
        actual, indices = {}, []
        for i in range(len(dataset)):
            info = dataset.get_data_info(i)
            ann_id, img_id = int(info["id"]), int(info["img_id"])
            if ann_id in actual:
                raise ValueError("Dataset duplicated an annotation")
            actual[ann_id] = img_id
            if ann_id in remaining:
                indices.append(i)
        if actual != expected:
            raise ValueError("Dataset silently filtered or changed approved manifest; resolve with L")
        model = init_model(cfg, str(args.checkpoint), device=args.device)
        model.requires_grad_(False)
        model.eval()
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=pseudo_collate)
        checked = False
        with torch.inference_mode():
            for batch in loader:
                if model.training or model.test_cfg.get("flip_test", False):
                    raise ValueError("Model protocol changed during caching")
                processed = model.data_preprocessor(batch, training=False)
                samples = processed["data_samples"]
                state = model.head.extract_refinement_state(model.extract_feat(processed["inputs"]))
                results = decode_original(model, state["p0_simcc"], samples)
                arrays = pack_batch(state, results, expected, images)
                if not checked:
                    # Exercise real serialization before committing any batch.
                    from io import BytesIO
                    buffer = BytesIO()
                    np.savez(buffer, **arrays)
                    buffer.seek(0)
                    with np.load(buffer, allow_pickle=False) as saved:
                        reloaded = {k: saved[k] for k in arrays}
                    check_reloaded(model, reloaded, samples, state, args.device)
                    checked = True
                    print("PASS: reloaded P0 and mixed-action candidate match live path")
                writer.append(arrays)
                print(f"Cached {len(writer.seen)}/{len(expected)}", flush=True)
        cache.verify(True)
        print(f"PASS: {protocol['scope']} cache complete ({len(expected)} instances)")


if __name__ == "__main__":
    main()
