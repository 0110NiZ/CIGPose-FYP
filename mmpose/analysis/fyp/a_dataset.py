"""Streaming full-train inputs plus Y's labels; no candidate/label generation."""
import argparse
from pathlib import Path

import numpy as np

from a_contract import GainLabels, label_metadata_template, model_inputs, require_train_cache
from shared_cache import ReferenceCache, atomic_json


class ATrainingData:
    def __init__(self, cache_directory, label_directory):
        self.cache = ReferenceCache(cache_directory)
        require_train_cache(self.cache)
        self.cache.verify(require_complete=True)
        self.labels = GainLabels(label_directory, self.cache)

    def batches(self, batch_size=256, seed=42, epoch=0, buffer_shards=32):
        """Shuffle shards then rows within bounded buffers; never drop the tail."""
        if batch_size < 1 or buffer_shards < 1:
            raise ValueError('batch_size/buffer_shards must be positive')
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        paths = self.cache.shards()
        rng.shuffle(paths)
        for start in range(0, len(paths), buffer_shards):
            chunks = [self.cache.load_shard(p) for p in paths[start:start + buffer_shards]]
            keys = ('features', 'uncertainty', 'P_0_coords', 'P_0_scores',
                    'input_center', 'input_scale', 'annotation_id', 'image_id')
            a = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
            order = rng.permutation(len(a['annotation_id']))
            for begin in range(0, len(order), batch_size):
                idx = order[begin:begin + batch_size]
                b = {k: v[idx] for k, v in a.items()}
                yield (model_inputs(b), self.labels.lookup(b['annotation_id'], b['image_id']),
                       b['annotation_id'])


def fit_input_statistics(cache):
    """Scalar uncertainty/score normalization, fitted on train only, saved in A."""
    require_train_cache(cache)
    total = 0
    sums, squares = np.zeros(2, np.float64), np.zeros(2, np.float64)
    for path in cache.shards():
        a = cache.load_shard(path)
        for i, k in enumerate(('uncertainty', 'P_0_scores')):
            v = a[k].astype(np.float64)
            sums[i] += v.sum()
            squares[i] += np.square(v).sum()
        total += a['uncertainty'].size
    if total != len(cache.expected) * 17:
        raise ValueError('Incomplete training inputs for normalization')
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache', type=Path, required=True)
    p.add_argument('--labels', type=Path)
    p.add_argument('--write-contract', type=Path,
                   help='Write a metadata TEMPLATE for Y, not labels or a completion claim')
    args = p.parse_args()
    cache = ReferenceCache(args.train_cache)
    require_train_cache(cache)
    if args.write_contract:
        if args.write_contract.exists():
            raise FileExistsError(args.write_contract)
        args.write_contract.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.write_contract, label_metadata_template(cache))
        print('Wrote unfinished label metadata template:', args.write_contract)
    if args.labels:
        data = ATrainingData(args.train_cache, args.labels)
        print('PASS full input/label identity checks:', len(data.cache.expected))
    if not args.labels and not args.write_contract:
        p.error('Use --labels to check Y delivery or --write-contract to prepare its interface')


if __name__ == '__main__':
    main()
