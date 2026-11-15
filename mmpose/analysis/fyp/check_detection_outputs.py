"""Offline sample-only GT pairing, independent OKS check and visual overlays.

Pairing is solely diagnostic, not used to filter formal AP/AR predictions.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.cocoeval import COCOeval
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from detection_cache import ReferenceCache
from oks_utility import compute_instance_oks, compute_gain
from shared_cache import atomic_json


def iou(box, gt):
    x,y,w,h = gt
    b = np.array([x,y,x+w,y+h])
    size = np.maximum(0, np.minimum(box[2:],b[2:])-np.maximum(box[:2],b[:2]))
    inter = size.prod()
    return inter / (np.prod(box[2:]-box[:2])+w*h-inter)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--ann',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    cache=ReferenceCache(a.cache); cache.verify(True)
    coco=json.loads(a.ann.read_text())
    images={r['id']:r for r in coco['images']}
    anns={}
    for r in coco['annotations']:
        if not r.get('iscrowd') and r['num_keypoints']>0 and r['area']>0:
            anns.setdefault(r['image_id'],[]).append(r)
    rows=[]; tiles=[]
    for path in cache.shards():
        data=cache.load_shard(path)
        for i,idx in enumerate(data['source_detection_index']):
            img_id=int(data['image_id'][i]); candidates=anns.get(img_id,[])
            ranked=sorted([(iou(data['bbox'][i],r['bbox']),r) for r in candidates],key=lambda x:x[0],reverse=True)
            match=ranked[0][1] if ranked and ranked[0][0]>=.5 else None
            rec=dict(detection_instance_id=cache.instance_id(idx),image_id=img_id,
                     diagnostic_match_iou=float(ranked[0][0]) if ranked else None,
                     gt_annotation_id=match['id'] if match else None)
            if match:
                gt=np.asarray(match['keypoints']).reshape(17,3)
                oks=[]
                for key in ('P_0_coords','top1_coords'):
                    pred=data[key][i]
                    value=compute_instance_oks(pred,gt,match['area'])
                    ev=COCOeval(iouType='keypoints')
                    ev._gts[(img_id,1)]=[match]
                    ev._dts[(img_id,1)]=[dict(score=1.,keypoints=np.column_stack([pred,np.ones(17)]).reshape(-1).tolist())]
                    reference=float(ev.computeOks(img_id,1)[0,0])
                    np.testing.assert_allclose(value,reference,rtol=0,atol=1e-12)
                    oks.append(value)
                gain=compute_gain(data['top1_coords'][i],data['P_0_coords'][i],gt,match['area'])
                assert abs(gain-(oks[1]-oks[0]))<1e-12
                rec.update(oks_p0=oks[0],oks_pc=oks[1],gain=gain,official_oks_matches=True)
            im=Image.open(a.images/images[img_id]['file_name']).convert('RGB')
            scale=min(1.,560/im.width); im=im.resize((int(im.width*scale),int(im.height*scale)))
            draw=ImageDraw.Draw(im)
            box=(data['bbox'][i]*scale).tolist();draw.rectangle(box,outline='yellow',width=2)
            if match:
                for x,y,v in gt:
                    if v>0: draw.ellipse((x*scale-4,y*scale-4,x*scale+4,y*scale+4),outline='lime',width=2)
            for field,color in [('P_0_coords','cyan'),('top1_coords','magenta')]:
                for x,y in data[field][i]*scale:
                    draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            tile=Image.new('RGB',(560,460),'white');tile.paste(im,(0,30))
            ImageDraw.Draw(tile).text((5,5),f'det {int(idx)} image {img_id} GT green / P0 cyan / Pc pink',fill='black')
            tile.save(a.output/f'detection-{int(idx):06d}.png');tiles.append(tile);rows.append(rec)
    sheet=Image.new('RGB',(1120,460*((len(tiles)+1)//2)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*560,(i//2)*460))
    sheet.save(a.output/'contact_sheet.png')
    assert any(r.get('official_oks_matches') for r in rows)
    atomic_json(a.output/'diagnostic_report.json',dict(passed=True,scope='sample-only offline diagnostics',
                matching_rule='independent highest bbox IoU >= 0.5; not a training matcher or evaluation filter',rows=rows))
    print('PASS: independent OKS/gain and visualization export',len(rows))


if __name__=='__main__':main()
