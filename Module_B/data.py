"""B inputs, label joins, image split and development metrics."""
import json
import hashlib
from pathlib import Path
import numpy as np
from Module_B.config import ALPHAS, FEATURE_SPEC
from Module_B.common import sha, digest

def image_number(image_id, seed, purpose):
    return int.from_bytes(hashlib.sha256(
        f'b-train-v1:{seed}:{purpose}:{int(image_id)}'.encode()).digest()[:8], 'big')



def is_development(image_id, seed):
    return image_number(image_id, seed, 'split') < (1 << 64)//10



def select_ids(expected, seed, max_instances):
    """A smoke limit retains whole images; selection and split use distinct salts."""
    groups = {}
    for ann_id, image_id in expected.items():
        groups.setdefault(image_id, []).append(ann_id)
    if max_instances is None:
        return set(expected)
    selected = set()
    for image_id in sorted(groups, key=lambda v: (image_number(v, seed, 'smoke'), v)):
        selected.update(groups[image_id])
        if len(selected) >= max_instances:
            break
    return selected



def make_features(arrays):
    """Shared inference input builder: this function never reads labels or GT."""
    f = arrays['features']
    joint = arrays['top1_joint']
    n = len(joint)
    if f.shape != (n, 17, 256) or not np.issubdtype(joint.dtype, np.integer):
        raise ValueError('Bad feature/joint schema')
    if np.any((joint < 0) | (joint >= 17)):
        raise ValueError('Joint index out of range')
    x = np.concatenate([
        f[np.arange(n), joint], f.mean(axis=1), arrays['uncertainty'],
        arrays['confidence'], np.eye(17, dtype=np.float32)[joint],
    ], axis=1).astype(np.float32)
    if x.shape != (n, FEATURE_SPEC['dimension']) or not np.isfinite(x).all():
        raise ValueError('Invalid B input')
    return x



def prepare_data(cache, label_dir, seed, max_instances, split_mode='internal'):
    if split_mode not in ('internal', 'all'):
        raise ValueError('Unknown training split mode')
    run = json.loads((label_dir/'run.json').read_text())
    summary = json.loads((label_dir/'summary.json').read_text())
    if (run.get('schema_version') != 'b-only-labels-v4' or run.get('scope') != 'full'
            or summary.get('scope') != 'full' or summary.get('passed') is not True
            or summary.get('fingerprint') != digest(run)
            or run.get('alphas') != ALPHAS
            or summary.get('instances') != 149813
            or summary.get('candidate_records') != 599252
            or run.get('source_cache_run_sha256') != sha(cache.directory/'run.json')
            or cache.run['protocol'].get('dataset_split') != 'train2017'):
        raise ValueError('Require the complete verified v4 train2017 label run and its source cache')
    infos = run['source_shards']
    ids = [int(i) for info in infos for i in info['annotation_ids']]
    if len(ids) != len(set(ids)) or set(ids) != set(cache.expected) or len(ids) != 149813:
        raise ValueError('Label manifest does not exactly cover train cache IDs')
    selected = select_ids(cache.expected, seed, max_instances)
    n = len(selected)
    x = np.empty((n, FEATURE_SPEC['dimension']), np.float32)
    gains = np.empty((n, 4), np.float64)
    ann_ids = np.empty(n, np.int64)
    image_ids = np.empty(n, np.int64)
    seen = set()
    offset = 0
    used_files = []
    fingerprint = digest(run)
    for seq, info in enumerate(infos, 1):
        if not selected.intersection(info['annotation_ids']):
            continue
        source = cache.directory/info['name']
        if Path(info['name']).name != info['name'] or sha(source) != info['sha256']:
            raise ValueError('Source shard hash/name mismatch')
        a = cache.load_shard(source)
        np.testing.assert_array_equal(a['annotation_id'], info['annotation_ids'])
        stem = source.stem
        report_path, payload = label_dir/(stem+'.json'), label_dir/(stem+'.npz')
        rep = json.loads(report_path.read_text())
        if (rep['fingerprint'] != fingerprint or rep['source'] != info
                or sha(payload) != rep['result_sha256']
                or not rep['source_arrays_unchanged']
                or not rep['alpha1_full_replacement_features_equal']):
            raise ValueError('Label shard provenance/hash/check mismatch: '+stem)
        with np.load(payload, allow_pickle=False) as labels:
            np.testing.assert_array_equal(labels['annotation_id'], a['annotation_id'])
            np.testing.assert_array_equal(labels['image_id'], a['image_id'])
            np.testing.assert_array_equal(labels['joint_id'], a['top1_joint'])
            np.testing.assert_array_equal(labels['alphas'], np.asarray(ALPHAS, np.float32))
            g = labels['gain'].copy()
            if g.shape != (len(a['annotation_id']), 4) or not np.isfinite(g).all():
                raise ValueError('Bad gain supervision')
            np.testing.assert_allclose(g, labels['oks_after']-labels['oks_before'][:, None],
                                       atol=1e-12, rtol=0)
        mask = np.asarray([int(v) in selected for v in a['annotation_id']])
        joined = set(map(int, a['annotation_id'][mask]))
        if seen & joined:
            raise ValueError('Duplicate training IDs')
        seen.update(joined)
        end = offset + int(mask.sum())
        x[offset:end] = make_features(a)[mask]
        gains[offset:end] = g[mask]
        ann_ids[offset:end], image_ids[offset:end] = a['annotation_id'][mask], a['image_id'][mask]
        offset = end
        used_files.append(dict(source=info['name'], source_sha256=info['sha256'],
            labels_sha256=rep['result_sha256'], report_sha256=sha(report_path)))
        if len(used_files) % 128 == 0:
            print(f'Loaded verified training rows: {offset}/{n}', flush=True)
    if seen != selected or offset != n:
        raise ValueError('Incomplete training join')
    if split_mode == 'all':
        dev = np.zeros(n, dtype=bool)
        print(f'Data ready: {n} people; all {n} used for training; internal development=0', flush=True)
        return x, gains, ann_ids, image_ids, dev, used_files, run, summary
    dev = np.asarray([is_development(v, seed) for v in image_ids])
    if not dev.any() or dev.all():
        raise ValueError('Empty train or development partition; increase smoke sample')
    if set(image_ids[dev]) & set(image_ids[~dev]):
        raise ValueError('Image leakage across train/development')
    print(f'Data ready: {n} people; train={(~dev).sum()}, development={dev.sum()}', flush=True)
    return x, gains, ann_ids, image_ids, dev, used_files, run, summary



def normalization(x, train_mask):
    mean = x[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.
    return mean, std



def gain_metrics(predicted, actual, fixed_from_train):
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError('Invalid B prediction')
    choice = predicted.argmax(axis=1)  # Exact predicted ties: smallest alpha.
    chosen = actual[np.arange(len(choice)), choice]
    best = actual.max(axis=1)
    return dict(selected_mean_gain=float(chosen.mean()),
        oracle_mean_gain=float(best.mean()), mean_oracle_regret=float((best-chosen).mean()),
        mean_gain_vs_fixed_from_train=float(chosen.mean()-actual[:, fixed_from_train].mean()),
        fixed_from_train_alpha=ALPHAS[fixed_from_train],
        fixed_alpha_mean_gains={str(v):float(actual[:,j].mean()) for j,v in enumerate(ALPHAS)},
        selected_alpha_counts={str(v):int((choice==j).sum()) for j,v in enumerate(ALPHAS)},
        selected_negative_people=int((chosen<0).sum()),
        all_candidate_negative_people=int((best<0).sum()),
        exact_best_tie_people=int(((actual==best[:,None]).sum(axis=1)>1).sum()))

