"""Extend a frozen COCO-30% split to 50% using image-level stratification."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'val', 'ids30', 'manifest30', 'annotations30', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    manifest30 = json.loads(args.manifest30.read_text())
    assert sha(args.train) == manifest30['source_sha256']['train']
    assert sha(args.val) == manifest30['source_sha256']['val']
    assert sha(args.annotations30) == manifest30['subset_sha256']
    raw = json.loads(args.train.read_text())
    val = json.loads(args.val.read_text())
    previous = json.loads(args.annotations30.read_text())
    old = set(json.loads(args.ids30.read_text()))
    assert old == {x['id'] for x in previous['images']} and len(old) == 16983
    images = {im['id']: im for im in raw['images']}
    by_image = defaultdict(list)
    valid = {}
    for ann in raw['annotations']:
        if 'bbox' not in ann or 'keypoints' not in ann:
            continue
        im = images[ann['image_id']]
        x, y, w, h = ann['bbox']
        box = np.array([np.clip(x, 0, im['width'] - 1),
                        np.clip(y, 0, im['height'] - 1),
                        np.clip(x + w, 0, im['width'] - 1),
                        np.clip(y + h, 0, im['height'] - 1)], dtype=np.float32)
        xyv = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        if (ann.get('iscrowd', 0) or ann['num_keypoints'] == 0
                or box[2] - box[0] <= 0 or box[3] - box[1] <= 0
                or xyv[:, :2].max() <= 0):
            continue
        valid[ann['id']] = ann
        by_image[ann['image_id']].append(ann)
    assert len(valid) == 149813 and len(by_image) == 56599
    strata = defaultdict(list)
    for iid, anns in sorted(by_image.items()):
        vs = np.concatenate([np.asarray(a['keypoints']).reshape(-1, 3)[:, 2] for a in anns])
        occ = float((vs == 1).sum() / max(1, (vs > 0).sum()))
        area = float(np.median([a['area'] for a in anns]))
        oc = 'low' if occ <= .2 else ('medium' if occ <= .5 else 'high')
        size = 'small' if area < 32**2 else ('medium' if area < 96**2 else 'large')
        n = len(anns)
        people = 'one' if n == 1 else ('two_to_four' if n <= 4 else 'five_plus')
        strata['|'.join((oc, size, people))].append(iid)
    for row in manifest30['strata']:
        ids = strata[row['stratum']]
        assert len(ids) == row['images_all']
        assert len(old.intersection(ids)) == row['images_selected']
        assert sum(len(by_image[i]) for i in ids) == row['instances_all']
    rng = np.random.default_rng(21)
    best = None
    for trial in range(64):
        chosen = set(old)
        for key, ids in sorted(strata.items()):
            count = int(np.floor(len(ids) * .5 + .5))
            remaining = sorted(set(ids) - old)
            extra = count - len(old.intersection(ids))
            assert 0 <= extra <= len(remaining)
            chosen.update(int(i) for i in rng.choice(remaining, extra, replace=False))
        people = sum(len(by_image[i]) for i in chosen)
        candidate = (abs(people - len(valid) * .5), trial, chosen, people)
        if best is None or candidate[0] < best[0]:
            best = candidate
    _, trial, chosen, people = best
    assert old <= chosen and chosen.isdisjoint(im['id'] for im in val['images'])
    subset = {k: v for k, v in raw.items() if k not in ('images', 'annotations')}
    subset['images'] = [im for im in raw['images'] if im['id'] in chosen]
    subset['annotations'] = [ann for ann in raw['annotations']
                             if ann['id'] in valid and ann['image_id'] in chosen]
    assert len(subset['annotations']) == people
    ann_map = {a['id']: a for a in subset['annotations']}
    assert all(ann_map[a['id']] == a for a in previous['annotations'])
    assert len(previous['annotations']) == 44941
    assert abs(people / len(valid) - .5) < .001
    annotation = args.out / 'person_keypoints_train2017_stratified50_nested30_seed21.json'
    write(annotation, subset)
    write(args.out / 'selected_image_ids.json', sorted(chosen))
    write(args.out / 'preserved30_image_ids.json', sorted(old))
    write(args.out / 'added_image_ids.json', sorted(chosen - old))
    audit = []
    for key, ids in sorted(strata.items()):
        selected = chosen.intersection(ids)
        audit.append(dict(stratum=key, images_all=len(ids), images30=len(old.intersection(ids)),
                          images50=len(selected), instances_all=sum(len(by_image[i]) for i in ids),
                          instances50=sum(len(by_image[i]) for i in selected)))
    summary = dict(seed=21, candidates=64, chosen_candidate_zero_based=trial,
                   eligible_images=len(by_image), full_valid_instances=len(valid),
                   selected_images=len(chosen), selected_valid_instances=people,
                   preserved_images=len(old), preserved_instances=44941,
                   added_images=len(chosen-old), added_instances=people-44941,
                   image_fraction=len(chosen)/len(by_image), instance_fraction=people/len(valid),
                   sampling='Preserve all frozen 30% images; reach nearest 50% image count in each stratum; select among 64 candidates by distance to 50% valid instances.',
                   source_sha256=manifest30['source_sha256'], source30_sha256=manifest30['subset_sha256'],
                   subset_sha256=sha(annotation), strata_definition=manifest30['stratum_definition'],
                   validation='Full unchanged COCO val2017; all selected training image IDs disjoint.',
                   strata=audit)
    write(args.out / 'manifest.json', summary)
    shutil.copyfile(__file__, args.out / Path(__file__).name)
    readme = (
        '# COCO 50% 固定训练子集\n\n'
        f'保留原 30% 的全部 {len(old):,} 张图片和 44,941 个人体实例，按原有遮挡、人体大小、人数分层补抽。\n\n'
        f'最终 {len(chosen):,} 张图片、{people:,} 个有效人体实例，人体比例 {people/len(valid):.6%}。\n\n'
        '训练时将子集标注放到 data/coco/annotations/，把训练数据集 ann_file 指向此文件。'
        '图片仍读取官方 COCO train2017/。验证继续读取完整 val2017 及官方验证标注。\n\n'
        'selected_image_ids.json 是所有入选图片 ID，preserved30_image_ids.json 是保留名单，'
        'added_image_ids.json 是补入名单。manifest.json 记录逐层数量和原始/子集 SHA256。'
        'SHA256SUMS.txt 可用于核对子集包内容。\n\n'
        '本包提供固定名单、标注、复现脚本和校验信息，组员配合官方 COCO 图片使用。\n\n'
        '复现脚本需要 Python 和 NumPy，执行示例为\n\n'
        '```bash\npython extend_coco30_to50.py --train person_keypoints_train2017.json '
        '--val person_keypoints_val2017.json --ids30 original30/selected_image_ids.json '
        '--manifest30 original30/manifest.json --annotations30 '
        'original30/person_keypoints_train2017_stratified30_seed21.json --out new50\n```\n'
    )
    (args.out / 'README.md').write_text(readme, encoding='utf-8')
    sums = ''.join(sha(f) + '  ' + f.name + '\n' for f in sorted(args.out.iterdir()) if f.is_file())
    (args.out / 'SHA256SUMS.txt').write_text(sums, encoding='utf-8')
    archive = args.out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(args.out.iterdir()):
            z.write(f, args.out.name + '/' + f.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        decoded = json.loads(z.read(args.out.name + '/' + annotation.name))
        assert len(decoded['images']) == len(chosen) and len(decoded['annotations']) == people
    archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('strata', 'strata_definition')}, indent=2))
    print('ZIP', archive, archive.stat().st_size, 'SHA256', sha(archive), flush=True)


if __name__ == '__main__':
    main()
