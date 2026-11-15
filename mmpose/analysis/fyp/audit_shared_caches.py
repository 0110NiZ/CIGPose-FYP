"""Verify full train GT / val detection identities, sources and byte-level shards."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from shared_cache import ReferenceCache as GTCache, atomic_json, hash_file
from detection_cache import ReferenceCache as DetectionCache
from read_val_detection_boxes import load_detection_instances


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache',type=Path,required=True)
    p.add_argument('--val-cache',type=Path)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    train=GTCache(a.train_cache); seen=train.verify(True)
    manifest=json.loads((ROOT/'manifests/train2017_valid_persons.json').read_text())
    train_ann=a.data/'annotations/person_keypoints_train2017.json'
    val_ann=a.data/'annotations/person_keypoints_val2017.json'
    assert hash_file(train_ann)==manifest['source_annotation_sha256']
    assert seen==set(manifest['valid_ann_ids'])
    source=json.loads(train_ann.read_text());val_source=json.loads(val_ann.read_text())
    gt={x['id']:x for x in source['annotations']}
    for idx,img in train.expected.items():assert gt[idx]['image_id']==img
    train_images={x['id'] for x in source['images']};val_images={x['id'] for x in val_source['images']}
    assert not train_images.intersection(val_images)
    report=dict(passed=True,train_count=len(seen),train_manifest_match=True,
                original_train_val_image_sets_disjoint=True,
                train_shards_validated=len(train.shards()),train_annotation_sha256=hash_file(train_ann))
    if a.val_cache:
        val=DetectionCache(a.val_cache); val_seen=val.verify(True)
        meta,records=load_detection_instances(a.data/'person_detection_results/COCO_val2017_detections_AP_H_56_person.json',val_ann,a.data/'val2017')
        assert val_seen==set(range(len(records)))
        for path in val.shards():
            batch=val.load_shard(path)
            for i,idx in enumerate(batch['source_detection_index']):
                r=records[int(idx)]
                assert int(batch['image_id'][i])==r['image_id']
                assert val.instance_id(idx)==r['detection_instance_id']
                np.testing.assert_array_equal(batch['bbox'][i],np.asarray(r['bbox_xyxy'],np.float32))
                assert batch['bbox_score'][i]==np.float32(r['bbox_score'])
                assert (a.data/'val2017'/r['image_file_name']).is_file()
        assert not set(train.expected.values()).intersection(val.expected.values())
        report.update(val_count=len(val_seen),val_images=len(set(val.expected.values())),
            val_shards_validated=len(val.shards()),all_detection_ids_boxes_scores_match=True,
            all_image_paths_exist=True,source_detection_sha256=meta['source_detection_sha256'])
    a.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(a.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
