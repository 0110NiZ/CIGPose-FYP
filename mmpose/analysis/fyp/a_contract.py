"""N/Y contract for Module A. Reads/checks labels; never generates gain labels."""
import json
from pathlib import Path

import numpy as np

from shared_cache import digest_json, hash_file

LABEL_SCHEMA = 'cigpose-module-a-labels-v1'
INPUT_SCHEMA = 'cigpose-module-a-inputs-v1'
CHECKPOINT_SCHEMA = 'cigpose-module-a-mlp-v1'
JOINT_ORDER = list(range(17))
GAIN_DEFINITION = 'OKS(P_j_alpha1,GT)-OKS(P_0,GT)'
INPUT_KEYS = ('features', 'uncertainty', 'pose_coords', 'pose_scores')


def require_train_cache(cache):
    p = cache.run['protocol']
    if (p.get('dataset_split') != 'train2017' or p.get('bbox_source') != 'gt'
            or p.get('scope') != 'full' or p.get('flip_test') is not False
            or p.get('flip_protocol') != 'single_view'
            or p.get('base_model_eval') is not True):
        raise ValueError('A training requires full train2017 GT, frozen eval, single-view cache')


def label_metadata_template(cache):
    """Fill provenance from an existing cache. Y fills shards/count/completion."""
    require_train_cache(cache)
    p = cache.run['protocol']
    return dict(schema_version=LABEL_SCHEMA, dataset_split='train2017',
                source_cache_fingerprint=cache.fingerprint,
                checkpoint_sha256=p['checkpoint_sha256'],
                annotation_sha256=p['annotation_sha256'],
                joint_order=JOINT_ORDER, alpha=1.0, flip_test=False,
                coordinate_system='original_image', gain_definition=GAIN_DEFINITION,
                expected_count=len(cache.expected), count=0, complete=False,
                generator_git_commit='', oks_implementation_sha256='', shards=[])


def validate_label_arrays(a):
    fields = {'image_id', 'annotation_id', 'gain', 'oks_before', 'oks_after'}
    if set(a) != fields:
        raise ValueError('Label shard must contain exactly: ' + ', '.join(sorted(fields)))
    if a['annotation_id'].ndim != 1:
        raise ValueError('annotation_id must be [N]')
    n = len(a['annotation_id'])
    if not n:
        raise ValueError('Empty label shard')
    for k in ('image_id', 'annotation_id'):
        if a[k].shape != (n,) or a[k].dtype != np.int64:
            raise ValueError(k + ' must be int64 [N]')
    if len(np.unique(a['annotation_id'])) != n:
        raise ValueError('Duplicate annotation ID within label shard')
    for k, shape in [('gain', (n, 17)), ('oks_before', (n,)), ('oks_after', (n, 17))]:
        if a[k].shape != shape or a[k].dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError(k + ' has incorrect shape/dtype')
        if not np.isfinite(a[k]).all():
            raise ValueError(k + ' contains nonfinite values')
    for k in ('oks_before', 'oks_after'):
        if np.any((a[k] < 0) | (a[k] > 1)):
            raise ValueError(k + ' must be within [0, 1]')
    if np.any(np.abs(a['gain']) > 1):
        raise ValueError('gain must be within [-1, 1]')
    if not np.allclose(a['gain'], a['oks_after'] - a['oks_before'][:, None],
                       atol=2e-6, rtol=0):
        raise ValueError('gain != oks_after - oks_before')


class GainLabels:
    """Small in-memory target table, joined by annotation ID, never by row order."""
    def __init__(self, directory, cache):
        require_train_cache(cache)
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        expected = label_metadata_template(cache)
        for k in ('schema_version', 'dataset_split', 'source_cache_fingerprint',
                  'checkpoint_sha256', 'annotation_sha256', 'joint_order', 'alpha',
                  'flip_test', 'coordinate_system', 'gain_definition', 'expected_count'):
            if self.metadata.get(k) != expected[k]:
                raise ValueError('Label metadata mismatch: ' + k)
        if self.metadata.get('complete') is not True or self.metadata.get('count') != len(cache.expected):
            raise ValueError('Training requires complete full-train labels')
        # A local rebuild may be identified by an archived, hashed source file.
        # Do not pretend that an uncommitted generator belongs to a base Git commit.
        snapshot = self.metadata.get('generator_source_snapshot')
        if snapshot:
            relative = Path(snapshot)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid generator source snapshot path')
            source = (self.directory / relative).resolve()
            if self.directory.resolve() not in source.parents:
                raise ValueError('Generator source snapshot escaped label directory')
            if hash_file(source) != self.metadata.get('generator_source_sha256'):
                raise ValueError('Generator source snapshot checksum mismatch')
        elif not self.metadata.get('generator_git_commit'):
            raise ValueError('Missing generator provenance: commit or verified source snapshot')
        if not self.metadata.get('oks_implementation_sha256'):
            raise ValueError('Missing OKS provenance')
        if 'precision_profile' in self.metadata:
            from precision_policy import require_profile
            require_profile(self.metadata['precision_profile'], self.metadata.get('precision_profile_sha256'))
        oks_hash = hash_file(Path(__file__).with_name('oks_utility.py'))
        if self.metadata['oks_implementation_sha256'] != oks_hash:
            raise ValueError('Y must use the shared oks_utility.py implementation')
        entries = self.metadata.get('shards', [])
        names = [e['file'] for e in entries]
        if (not names or len(set(names)) != len(names)
                or any(Path(n).name != n or '/' in n or '\\' in n or not n.endswith('.npz') for n in names)):
            raise ValueError('Expected unique label shard basenames ending in .npz')
        if set(names) != {p.name for p in self.directory.glob('*.npz')}:
            raise ValueError('Label manifest and NPZ files differ')
        chunks, seen = [], set()
        for e in entries:
            path = self.directory / e['file']
            if hash_file(path) != e['sha256']:
                raise ValueError('Label checksum mismatch: ' + e['file'])
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            validate_label_arrays(a)
            ids = a['annotation_id'].tolist()
            if e['count'] != len(ids) or seen.intersection(ids):
                raise ValueError('Label count mismatch or duplicate IDs across shards')
            for ann_id, img_id in zip(ids, a['image_id'].tolist()):
                if cache.expected.get(ann_id) != img_id:
                    raise ValueError('Label image/person does not match reference cache')
            seen.update(ids)
            chunks.append(a)
        if seen != set(cache.expected):
            raise ValueError('Missing/unexpected training labels')
        ids = np.concatenate([a['annotation_id'] for a in chunks])
        order = np.argsort(ids)
        self.ids = ids[order]
        self.images = np.concatenate([a['image_id'] for a in chunks])[order]
        self.gains = np.concatenate([a['gain'] for a in chunks])[order].astype(np.float32)
        self.fingerprint = digest_json(self.metadata)

    def lookup(self, annotation_ids, image_ids):
        ids = np.asarray(annotation_ids)
        positions = np.searchsorted(self.ids, ids)
        if np.any(positions >= len(self.ids)) or not np.array_equal(self.ids[positions], ids):
            raise ValueError('Requested annotation has no label')
        if not np.array_equal(self.images[positions], image_ids):
            raise ValueError('Wrong image ID in label join')
        return self.gains[positions].copy()


def model_inputs(a):
    """Explicit whitelist: IDs, GT, gain and annotation area cannot enter A."""
    n = len(a['features'])
    specs = {'features': (n, 17, 256), 'uncertainty': (n, 17),
             'P_0_coords': (n, 17, 2), 'P_0_scores': (n, 17),
             'input_center': (n, 2), 'input_scale': (n, 2)}
    for k, shape in specs.items():
        if np.asarray(a[k]).shape != shape or not np.isfinite(a[k]).all():
            raise ValueError('Invalid input: ' + k)
    if np.any(a['input_scale'] <= 0):
        raise ValueError('input_scale must be positive')
    coords = (a['P_0_coords'] - a['input_center'][:, None, :]) / a['input_scale'][:, None, :]
    return dict(features=np.asarray(a['features'], np.float32),
                uncertainty=np.asarray(a['uncertainty'], np.float32),
                pose_coords=np.asarray(coords, np.float32),
                pose_scores=np.asarray(a['P_0_scores'], np.float32))
