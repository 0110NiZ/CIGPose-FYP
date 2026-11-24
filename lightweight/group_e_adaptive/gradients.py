"""FP16 gradient probes and common-factor search on fixed inputs."""
import copy
import math
import random
from itertools import islice

import numpy as np
import torch

from c_tracking import preserve_rng

LOSSES = ('loss_kpt', 'loss_kd', 'loss_feature', 'loss_cf')
LABELS = ('gt', 'simcc', 'feature', 'cf')
MIN_FACTOR, MAX_FACTOR = 0.01, 100.0
TARGET = 0.45
BATCH_COUNT = 5


def build_batches(runner, stage_one):
    cfg = copy.deepcopy(runner.cfg.train_dataloader)
    cfg.dataset.pipeline = copy.deepcopy(stage_one)
    cfg.update(num_workers=0, persistent_workers=False)
    cfg.pop('prefetch_factor', None)
    with preserve_rng():
        random.seed(runner.seed)
        np.random.seed(runner.seed)
        torch.manual_seed(runner.seed)
        loader = runner.build_dataloader(cfg, seed=runner.seed)
        return list(islice(loader, BATCH_COUNT))


def parameter_scope(name):
    for prefix, scope in (('backbone.', 'backbone'), ('head.cim.', 'cim'),
                          ('head.gcn.', 'gnn')):
        if name.startswith(prefix):
            return scope
    return 'other'


def batch_grams(losses, parameters, scopes):
    # The fifth gradient independently checks the sum of the first four.
    objective = [losses[key].float() for key in LOSSES]
    objective.append(sum(objective))
    for scale in (1024., 512., 256., 128., 64., 32., 16., 8., 4., 2., 1.):
        grads = [
            torch.autograd.grad(loss * scale, parameters, retain_graph=True,
                                allow_unused=True)
            if loss.requires_grad else (None,) * len(parameters)
            for loss in objective
        ]
        sums = torch.zeros((4, 5, 5), device=parameters[0].device, dtype=torch.float64)
        # One parameter tensor at a time keeps peak diagnostic memory bounded.
        for scope, parts in zip(scopes, zip(*grads, strict=True), strict=True):
            reference = next((part for part in parts if part is not None), None)
            if reference is None:
                continue
            matrix = torch.stack([
                part.detach().reshape(-1).double() / scale if part is not None
                else torch.zeros(reference.numel(), device=reference.device, dtype=torch.float64)
                for part in parts
            ])
            sums[scope].add_(matrix @ matrix.T)
        del grads
        grams = sums.cpu().numpy()
        if np.isfinite(grams).all():
            return scale, grams
    return None, None


def probe(runner, batches, base_weights):
    model = runner.model
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    names, parameters = zip(*named)
    scope_names = ('backbone', 'cim', 'gnn', 'other')
    scopes = [scope_names.index(parameter_scope(name)) for name in names]
    flags = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    current = {key: model.train_cfg[key] for key in base_weights}
    model.train_cfg.update(base_weights)
    records = []

    def restore_buffers():
        with torch.no_grad():
            for buffer, initial in buffers:
                buffer.copy_(initial)

    try:
        with preserve_rng(), torch.enable_grad():
            torch.manual_seed(runner.seed)
            model.train()
            for index, batch in enumerate(batches, 1):
                data = model.data_preprocessor(copy.deepcopy(batch), training=True)
                with runner.optim_wrapper.optim_context(model):
                    losses = model(**data, mode='loss')
                scale, grams = batch_grams(losses, parameters, scopes)
                del losses, data
                restore_buffers()
                record = dict(batch=index, loss_scale=scale)
                if grams is not None:
                    record['grams'] = dict(zip(scope_names, grams))
                    record['grams']['all'] = grams.sum(axis=0)
                records.append(record)
    finally:
        restore_buffers()
        for module, training in flags:
            module.training = training
        model.train_cfg.update(current)
    return records


def contribution(records, factor):
    if not records or any('grams' not in record for record in records):
        return None
    grams = np.array([record['grams']['all'][:4, :4] for record in records])
    values = contributions_for_grid(grams, [factor])
    return float(values[0]) if np.isfinite(values[0]) else None


def contributions_for_grid(grams, factors):
    kd = grams[:, 1:3, 1:3].sum(axis=(1, 2))[:, None]
    cross = grams[:, [0, 3], :][:, :, 1:3].sum(axis=(1, 2))[:, None]
    base = grams[:, [0, 3], :][:, :, [0, 3]].sum(axis=(1, 2))[:, None]
    f = np.asarray(factors)[None, :]
    denominator = base + 2 * f * cross + f * f * kd
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denominator > 0, (f * cross + f * f * kd) / denominator,
                        np.nan).mean(axis=0)


def choose_factor(records, old_factor, before):
    if before is None:
        return old_factor, dict(status='invalid_probe', candidates=[], hit_limit=False)
    if 0.40 <= before <= 0.50:
        return old_factor, dict(status='kept_in_range', candidates=[], hit_limit=False)

    grams = np.array([record['grams']['all'][:4, :4] for record in records])
    factors = np.unique(np.append(np.geomspace(MIN_FACTOR, MAX_FACTOR, 401), old_factor))
    values = contributions_for_grid(grams, factors)
    candidates = [[float(f), float(v) if np.isfinite(v) else None]
                  for f, v in zip(factors, values, strict=True)]
    for left, right in zip(candidates[:-1], candidates[1:], strict=True):
        lo, low_value = left
        hi, high_value = right
        if low_value is None or high_value is None or (low_value - TARGET) * (high_value - TARGET) >= 0:
            continue
        for _ in range(24):
            middle = math.sqrt(lo * hi)
            value = float(contributions_for_grid(grams, [middle])[0])
            if not math.isfinite(value):
                break
            if (value - TARGET) * (low_value - TARGET) > 0:
                lo, low_value = middle, value
            else:
                hi = middle
        f = math.sqrt(lo * hi)
        v = float(contributions_for_grid(grams, [f])[0])
        candidates.append([f, v if math.isfinite(v) else None])

    feasible = [(f, v) for f, v in candidates if v is not None and 0.40 <= v <= 0.50]
    if not feasible:
        return old_factor, dict(status='unreachable_kept_previous', candidates=candidates,
                                hit_limit=True)
    factor, _ = min(feasible, key=lambda item: (abs(item[1] - TARGET),
                                               abs(math.log(item[0] / old_factor))))
    return factor, dict(status='adjusted', candidates=candidates,
                        hit_limit=factor in (MIN_FACTOR, MAX_FACTOR))


def summarize_gram(gram, factor):
    # The independent direct-sum check refers to the original base coefficients.
    reference_norm = math.sqrt(max(0., gram[4, 4]))
    difference = gram[:4, :4].sum() + gram[4, 4] - 2 * gram[4, :4].sum()
    relative_error = math.sqrt(max(0., difference)) / reference_norm if reference_norm else None
    weights = np.array([1., factor, factor, 1.])
    weighted = gram[:4, :4] * np.outer(weights, weights)
    norms = np.sqrt(np.maximum(0., np.diag(weighted)))
    total_squared = float(weighted.sum())
    projection = weighted.sum(axis=1) / total_squared if total_squared > 0 else [None] * 4

    def cosine(i):
        denominator = norms[0] * norms[i]
        return float(np.clip(weighted[0, i] / denominator, -1, 1)) if denominator else None

    return dict(norms=dict(zip(LABELS, norms.tolist())),
                projection=dict(zip(LABELS, map(lambda x: float(x) if x is not None else None,
                                                projection))),
                gt_simcc_cosine=cosine(1), gt_feature_cosine=cosine(2),
                base_gradient_sum_relative_error=relative_error)
