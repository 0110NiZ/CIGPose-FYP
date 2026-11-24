"""Identity and runtime checks for the supplied fixed subset (no sampling)."""
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED_SHA256 = 'bce0685bc39a2086881b4c434e931637371b68d06c2502438d271873d5a90f46'
VERSION = 'COCO50_nested30_fixed_annotations_seed21'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_annotations(path):
    path = Path(path).resolve(strict=True)
    actual = digest(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(f'Not the supplied fixed 50% annotations: {path}; SHA256={actual}')
    data = json.loads(path.read_text(encoding='utf-8'))
    images = {int(row['id']) for row in data['images']}
    annotations = {int(row['id']): row for row in data['annotations']}
    if len(images) != len(data['images']) or len(annotations) != len(data['annotations']):
        raise ValueError('Duplicate image or annotation IDs')
    if any(int(row['image_id']) not in images for row in annotations.values()):
        raise ValueError('Annotation references an absent image')
    return dict(version=VERSION, path=str(path), sha256=actual,
                images=len(images), annotation_instances=len(annotations)), annotations


def inspect_loader(loader, annotation_path):
    """Read unaugmented dataset records; never iterate the sampler or training loader."""
    identity, annotations = inspect_annotations(annotation_path)
    dataset = loader.dataset
    if Path(dataset.ann_file).resolve() != Path(annotation_path).resolve():
        raise ValueError(f'Loaded dataset ann_file differs: {dataset.ann_file}')
    if getattr(loader.sampler, 'world_size', 1) != 1:
        raise ValueError('This experiment preserves the original single-GPU setup')
    if type(loader.sampler).__name__ not in ('DefaultSampler', 'RandomSampler', 'SequentialSampler'):
        raise ValueError('Unsupported sampler: cannot verify epoch population')
    if getattr(loader.sampler, 'replacement', False) or len(loader.sampler) != len(dataset):
        raise ValueError('Expected one pass through the actual dataset per epoch')
    if loader.batch_size != 128 or loader.drop_last:
        raise ValueError('Expected batch_size=128 and drop_last=False; do not silently drop people')
    loaded = set()
    for i in range(len(dataset)):
        info = dataset.get_data_info(i)
        aid = int(info['id'])
        if aid in loaded or aid not in annotations:
            raise ValueError(f'Duplicate or foreign loaded annotation ID: {aid}')
        raw = info.get('raw_ann_info')
        if raw is None or raw != annotations[aid]:
            raise ValueError(f'Loaded annotation differs from fixed JSON: {aid}')
        loaded.add(aid)
    batches = (len(dataset) + loader.batch_size - 1) // loader.batch_size
    if len(loader) != batches:
        raise ValueError('Actual dataloader length differs from sampler/batch-size calculation')
    report = dict(**identity, dataset_people=len(dataset), sampler_people=len(loader.sampler),
                  batches_per_epoch=len(loader), planned_train_iterations=100 * len(loader),
                  filtered_annotation_count=len(annotations) - len(loaded),
                  loaded_annotation_ids_sha256=hashlib.sha256(
                      json.dumps(sorted(loaded)).encode()).hexdigest(),
                  sampler=type(loader.sampler).__name__, batch_size=loader.batch_size,
                  drop_last=loader.drop_last)
    return report, loaded, annotations


def reuse_diagnostics(source, destination, manifest, loaded, annotations):
    """Verify provenance, recipe and GT, then copy the augmented input bytes verbatim."""
    import torch
    source = Path(source).resolve(strict=True)
    saved = json.loads((source.parent / 'run_manifest.json').read_text(encoding='utf-8'))
    if digest(source) != saved['diagnostic_sha256']:
        raise ValueError('30% diagnostic cache hash does not match its run_manifest.json')
    old, new = saved['recipe'], manifest['recipe']
    for key in ('seed', 'temperature', 'simcc_base_weight', 'feature_base_weight',
                'base_lr', 'batch_size', 'epochs', 'diagnostic_augmentation',
                'student_backbone', 'student_cim_gnn_initialization'):
        if old[key] != new[key]:
            raise ValueError(f'30% diagnostic recipe mismatch: {key}')
    if old['teacher']['sha256'] != new['teacher']['sha256']:
        raise ValueError('30% teacher checkpoint mismatch')
    if old['data']['val_annotations']['sha256'] != new['data']['val_annotations']['sha256']:
        raise ValueError('30% validation annotations mismatch')
    for key, value in new['sources'].items():
        if key.endswith(('/train.py', '/tracking.py', '/subset50.py')):
            continue
        if old['sources'].get(key) != value:
            raise ValueError(f'30% model/config/gradient source mismatch: {key}')
    batches = torch.load(source, map_location='cpu', weights_only=False)
    if len(batches) != 5:
        raise ValueError('Expected five diagnostic batches')
    for index, batch in enumerate(batches):
        samples = batch['data_samples']
        if len(samples) != 128 or len(batch['inputs']) != 128:
            raise ValueError('Each diagnostic batch must contain 128 augmented inputs')
        record = saved['diagnostic_batches'][index]
        if record['sample_ids'] != [str(s.id) for s in samples]:
            raise ValueError('Diagnostic sample IDs differ from source manifest')
        for sample in samples:
            aid = int(sample.id)
            if aid not in loaded:
                return None, dict(status='generated', reason=f'30% diagnostic sample {aid} absent from actual 50% dataset')
            if sample.metainfo.get('raw_ann_info') != annotations[aid]:
                raise ValueError(f'Diagnostic GT mismatch: {aid}')
            if int(sample.img_id) != int(annotations[aid]['image_id']):
                raise ValueError('Diagnostic image ID mismatch')
            if 'transformed_keypoints' not in sample.gt_instances:
                raise ValueError('Missing augmented diagnostic ground truth')
    shutil.copyfile(source, destination)
    if digest(destination) != saved['diagnostic_sha256']:
        raise RuntimeError('Diagnostic cache changed while copying')
    return batches, dict(status='reused_verbatim', source=str(source),
                         source_manifest_sha256=digest(source.parent / 'run_manifest.json'),
                         sha256=digest(destination))
