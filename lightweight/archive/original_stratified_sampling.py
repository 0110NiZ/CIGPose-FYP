import os,json,pathlib,hashlib,collections,copy,numpy as np,torch
from mmengine.config import Config
from mmpose.utils import register_all_modules
from mmpose.registry import DATASETS,MODELS
ROOT=pathlib.Path('/root/autodl-tmp/cigpose-original')
EXP=ROOT/'experiments/small_backbone_30_seed21_precision3'
EXP.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT/'CIGPose/mmpose')
register_all_modules()
base=Config.fromfile('projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py')
source=ROOT/'data/coco/annotations/person_keypoints_train2017.json'
val=ROOT/'data/coco/annotations/person_keypoints_val2017.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
hashes={'train':sha(source),'val':sha(val)}
dcfg=copy.deepcopy(base.train_dataloader.dataset)
dcfg.pipeline=[];dcfg.serialize_data=False
ds=DATASETS.build(dcfg)
assert len(ds)==149813,len(ds)
valid={x['id']:x for x in ds.data_list}
by_image=collections.defaultdict(list)
for x in ds.data_list:by_image[int(x['img_id'])].append(x['raw_ann_info'])
strata=collections.defaultdict(list)
rows={}
for iid,anns in sorted(by_image.items()):
    vs=np.concatenate([np.array(a['keypoints']).reshape(-1,3)[:,2] for a in anns])
    occ=float((vs==1).sum()/max(1,(vs>0).sum()))
    area=float(np.median([a['area'] for a in anns]))
    oc='low' if occ<=.2 else ('medium' if occ<=.5 else 'high')
    size='small' if area<32**2 else ('medium' if area<96**2 else 'large')
    n=len(anns);people='one' if n==1 else ('two_to_four' if n<=4 else 'five_plus')
    key='|'.join([oc,size,people]);strata[key].append(iid)
    rows[iid]={'image_id':iid,'stratum':key,'valid_instances':n,'occluded_labeled_keypoint_ratio':occ,'median_person_area':area}
rng=np.random.default_rng(21);target=len(ds)*.3;best=None
for trial in range(64):
    chosen=set()
    for key,ids in sorted(strata.items()):
        count=int(np.floor(len(ids)*.3+.5))
        chosen.update(int(x) for x in rng.choice(ids,count,replace=False))
    total=sum(rows[i]['valid_instances'] for i in chosen)
    if best is None or abs(total-target)<best[0]:best=(abs(total-target),chosen,total,trial)
_,chosen,total,trial=best
assert abs(total-target)/target<.02
raw=json.loads(source.read_text())
sub={k:v for k,v in raw.items() if k not in ['images','annotations']}
sub['images']=[i for i in raw['images'] if i['id'] in chosen]
sub['annotations']=[a for a in raw['annotations'] if a['id'] in valid and a['image_id'] in chosen]
assert len(sub['annotations'])==total
val_ids={x['id'] for x in json.loads(val.read_text())['images']}
assert chosen.isdisjoint(val_ids)
subpath=EXP/'person_keypoints_train2017_stratified30_seed21.json'
subpath.write_text(json.dumps(sub,separators=(',',':')))
(EXP/'selected_image_ids.json').write_text(json.dumps(sorted(chosen)))
summary={'seed':21,'eligible_images':len(by_image),'full_valid_instances':len(ds),'selected_images':len(chosen),'selected_valid_instances':total,'instance_fraction':total/len(ds),'image_fraction':len(chosen)/len(by_image),'sampling':'64 seeded candidates, each selecting nearest 30% of images within every stratum; choose closest to 30% valid instances','selected_candidate':trial,'stratum_definition':{'occlusion':'v=1 / count(v>0), aggregated per image; <=0.2, <=0.5, >0.5; v=0 excluded as unlabeled','size':'median annotated person area: <32^2, <96^2, >=96^2 pixels','people':'valid persons per image: 1, 2-4, >=5'},'source_sha256':hashes,'subset_sha256':sha(subpath),'validation':'unchanged full val2017 with original GT boxes','strata':[]}
for key,ids in sorted(strata.items()):
    picked=[i for i in ids if i in chosen]
    summary['strata'].append({'stratum':key,'images_all':len(ids),'images_selected':len(picked),'instances_all':sum(rows[i]['valid_instances'] for i in ids),'instances_selected':sum(rows[i]['valid_instances'] for i in picked)})
(EXP/'stratification_audit.json').write_text(json.dumps(summary,indent=2))
(EXP/'image_strata.json').write_text(json.dumps(list(rows.values())))
cfg=copy.deepcopy(base)
cfg.model.backbone.deepen_factor=.33;cfg.model.backbone.widen_factor=.5
cfg.model.head.in_channels=512
cfg.model.backbone.init_cfg.checkpoint=str(ROOT/'checkpoints/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth')
cfg.train_dataloader.dataset.ann_file=str(subpath)
cfg.train_dataloader.num_workers=4;cfg.val_dataloader.num_workers=2
cfg.max_epochs=100;cfg.stage2_num_epochs=7
cfg.train_cfg.max_epochs=100;cfg.train_cfg.val_interval=10
for sched in cfg.param_scheduler:
    if sched.type=='CosineAnnealingLR':sched.begin=50;sched.end=100;sched.T_max=50
for hook in cfg.custom_hooks:
    if hook.type=='mmdet.PipelineSwitchHook':hook.switch_epoch=93
cfg.work_dir=str(EXP/'work_dirs/train100')
cfg.launcher='none'
assert cfg.val_dataloader.dataset==base.val_dataloader.dataset
assert cfg.val_evaluator==base.val_evaluator
small=MODELS.build(cfg.model)
original=MODELS.build(base.model)
p_small=sum(p.numel() for p in small.parameters());p_original=sum(p.numel() for p in original.parameters())
assert round(p_small/1e6,3)==6.190,(p_small,p_original)
summary['model']={'parameters':p_small,'original_parameters':p_original,'reduction_percent':100*(1-p_small/p_original),'head_dim':256,'width':.5,'depth':.33}
summary['schedule']={'epochs':100,'lr':.001,'batch':384,'warmup_iterations':1000,'cosine_begin_epoch':50,'cosine_end_epoch':100,'stage2_switch_epoch':93}
cfg.dump(str(EXP/'cigpose_small30_100e.py'))
check=copy.deepcopy(cfg.train_dataloader.dataset);check.pipeline=[];check.serialize_data=False
subset_ds=DATASETS.build(check)
assert len(subset_ds)==total
assert hashes=={'train':sha(source),'val':sha(val)}
summary['checks']={'dataset_count':len(subset_ds),'train_val_disjoint':True,'original_annotations_unchanged':True,'validation_protocol_unchanged':True}
(EXP/'manifest.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='strata'},indent=2),flush=True)
