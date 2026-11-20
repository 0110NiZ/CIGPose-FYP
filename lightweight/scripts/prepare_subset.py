"""Rebuild the exact measured subset from frozen image IDs; never edit COCO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mmpose'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, default=ROOT / 'data/coco')
    args = parser.parse_args()
    coco = args.coco_root.resolve()
    dest = ROOT / 'lightweight/data'
    manifest = json.loads((dest / 'manifest.json').read_text())
    source = coco / 'annotations/person_keypoints_train2017.json'
    val = coco / 'annotations/person_keypoints_val2017.json'
    for label, path in [('train', source), ('val', val)]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['source_sha256'][label], f'Wrong {label} annotations'
    from mmengine.config import Config
    from mmpose.utils import register_all_modules
    from mmpose.registry import DATASETS
    os.chdir(ROOT / 'mmpose')
    register_all_modules()
    cfg = Config.fromfile(str(ROOT / 'lightweight/configs/team128_baseline100.py'))
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg.update(data_root=str(coco), ann_file=str(source), pipeline=[], serialize_data=False)
    ds = DATASETS.build(dataset_cfg)
    assert len(ds) == 149813
    valid = {item['id'] for item in ds.data_list}
    chosen = set(json.loads((dest / 'selected_image_ids.json').read_text()))
    raw = json.loads(source.read_text())
    subset = {k: v for k, v in raw.items() if k not in ['images', 'annotations']}
    subset['images'] = [i for i in raw['images'] if i['id'] in chosen]
    subset['annotations'] = [a for a in raw['annotations'] if a['id'] in valid and a['image_id'] in chosen]
    assert len(subset['images']) == 16983 and len(subset['annotations']) == 44941
    assert chosen.isdisjoint(i['id'] for i in json.loads(val.read_text())['images'])
    payload = json.dumps(subset, separators=(',', ':')).encode()
    assert hashlib.sha256(payload).hexdigest() == manifest['subset_sha256']
    target = dest / 'person_keypoints_train2017_stratified30_seed21.json'
    target.write_bytes(payload)
    print(f'Verified: {target}; 16983 images, 44941 valid people; original data unchanged.')

if __name__ == '__main__':
    main()
