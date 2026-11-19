"""Collect the aligned full-train / val-AP-selected experiment for GitHub."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from Module_B.config import LABELS, TRAINING, EVALUATION, EXPECTED_DETECTIONS
from Module_B.common import sha, digest, atomic_json, source_hashes
from Module_B.evaluate import validate_b_run, MODES
from Module_B.validate import PROTOCOL, TRAIN_FILES


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_reports(labels,training,evaluation,history):
    lr,ls=read_json(labels/'run.json'),read_json(labels/'summary.json')
    tr,ts=validate_b_run(training,Path(__file__).with_name('train.py'))
    er,es=read_json(evaluation/'run.json'),read_json(evaluation/'summary.json')
    if tr.get('protocol')!=PROTOCOL:
        raise ValueError('Use aligned AP training here; the original results folder is left unchanged')
    if (lr.get('scope')!='full' or ls.get('passed') is not True
            or ls.get('fingerprint')!=digest(lr) or ls.get('instances')!=149813
            or ls.get('candidate_records')!=599252
            or tr['labels_fingerprint']!=digest(lr)
            or tr['labels_summary_sha256']!=sha(labels/'summary.json')):
        raise ValueError('Label report/provenance mismatch')
    split=read_json(training/'split.json')
    if (tr['split_sha256']!=digest(split) or split.get('protocol')!=PROTOCOL
            or len(split['train_annotation_ids'])!=149813
            or len(set(split['train_annotation_ids']))!=149813
            or split['development_annotation_ids'] or split['development_image_ids']
            or len(split['validation_image_ids'])!=5000
            or set(split['train_image_ids'])&set(split['validation_image_ids'])):
        raise ValueError('Aligned train/validation membership mismatch')
    best=read_json(training/'best_validation.json')
    if (best['epoch']!=ts['best_epoch'] or best['metrics']!=ts['best_validation_metrics']
            or best['validation_AP']!=ts['best_validation_AP']):
        raise ValueError('Best validation recheck mismatch')
    if (er.get('scope')!='full' or es.get('scope')!='full'
            or es.get('formal_coco_ap_evaluated') is not True
            or es.get('prediction_checks_passed') is not True
            or es.get('fingerprint')!=digest(er)
            or er['b_training_fingerprint']!=digest(tr)
            or er['b_checkpoint_sha256']!=sha(training/'best.pt')
            or er['b_training_summary_sha256']!=sha(training/'summary.json')
            or er.get('b_model_selection_protocol')!=PROTOCOL
            or er.get('source_cache_run_sha256')!=tr['validation']['cache_run_sha256']
            or er.get('annotation_sha256')!=tr['validation']['annotation_sha256']
            or er.get('checkpoint_sha256')!=tr['validation']['checkpoint_sha256']
            or er.get('scoring')!=dict(score_mode='bbox_keypoint',keypoint_score_thr=.2,nms_mode='oks_nms',nms_thr=.9)
            or es.get('detections')!=EXPECTED_DETECTIONS
            or es.get('training_selection_AP_matches') is not True
            or es.get('val2017_used_for_model_selection') is not True
            or es.get('independent_test') is not False):
        raise ValueError('Aligned evaluation/model provenance mismatch')
    if set(es.get('evaluations',{}))!=set(MODES) or set(es.get('ap_0_to_1',{}))!=set(MODES):
        raise ValueError('Expected all four controls')
    if abs(es['ap_0_to_1']['b_current']-ts['best_validation_AP'])>1e-12:
        raise ValueError('Final B AP differs from selected validation AP')
    for mode,result in es['evaluations'].items():
        if (not result['all_10_metrics_match'] or not result['postprocessing_records_match']
                or result['audit']['evaluation_image_count']!=5000
                or result['input_count']!=EXPECTED_DETECTIONS
                or read_json(evaluation/mode/'metrics.json')!=result):
            raise ValueError('Final scoring consistency mismatch: '+mode)
    # Old labels retain their original source hash; the old archive is preserved.
    with zipfile.ZipFile(history) as archive:
        known=hashlib.sha256(archive.read('generate_b_labels_v4.py')).hexdigest()
    actual=[v for p,v in lr['code_sha256'].items() if Path(p).name=='generate_b_labels_v4.py']
    if actual!=[known]:raise ValueError('Original label source/archive mismatch')
    return es


def source_archive():
    folder=Path(__file__).resolve().parent
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.glob('*.py')):
            info=zipfile.ZipInfo('Module_B/'+path.name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
    return buffer.getvalue()


def collect(labels,training,evaluation,destination,history):
    for source in (labels,training,evaluation):
        if source==destination or source in destination.parents or destination in source.parents:
            raise ValueError('Copy reports to a separate new folder')
    summary=check_reports(labels,training,evaluation,history)
    plan=[]
    for source,subfolder,names in (
        (labels,'labels',('run.json','summary.json','resolved_config.py')),
        (training,'train',('run.json','summary.json','split.json','history.json','best.pt','best_validation.json')),
        (evaluation,'evaluate',('run.json','summary.json')),
    ):
        for name in names:
            path=source/name
            if not path.is_file():raise FileNotFoundError(path)
            plan.append((path,Path(subfolder)/name))
    for mode in MODES:
        plan.append((evaluation/mode/'metrics.json',Path('evaluate')/mode/'metrics.json'))
    source_bytes=source_archive()
    manifest=dict(protocol=PROTOCOL,files={str(r):sha(p) for p,r in plan},
        source_code=source_hashes(*TRAIN_FILES),source_archive_sha256=hashlib.sha256(source_bytes).hexdigest(),
        history_sha256=sha(history),original_directories=dict(labels=str(labels),train=str(training),evaluate=str(evaluation)),
        note='Original reports and weights copied byte for byte; original experiment remains in results/')
    if destination.exists():
        if read_json(destination/'files.json')!=manifest:raise FileExistsError('Different snapshot exists')
        for _,relative in plan:
            if sha(destination/relative)!=manifest['files'][str(relative)]:raise ValueError('Copied report changed')
        if sha(destination/'source.zip')!=manifest['source_archive_sha256']:raise ValueError('Source archive changed')
        print('Aligned reports already copied and verified:',destination)
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.results-',dir=destination.parent))
    try:
        for source,relative in plan:
            target=temp/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            if sha(target)!=manifest['files'][str(relative)]:raise ValueError('Report copy differs')
        (temp/'source.zip').write_bytes(source_bytes)
        (temp/'.gitattributes').write_text('* -text\n')
        atomic_json(temp/'files.json',manifest)
        names=dict(zip(MODES,('P0','Fixed alpha=0.25','Top-1 alpha=1','B')))
        lines=['# Module B aligned validation results','',
            'All 149813 approved train2017 people train B. No internal development holdout.',
            'Full val2017 COCO AP selects the best epoch. Exact ties retain the earlier epoch.',
            'This is selected validation performance, not an independent test score.','',
            '| Method | AP |','| --- | ---: |']
        for mode in MODES:lines.append(f"| {names[mode]} | {100*summary['ap_0_to_1'][mode]:.4f} |")
        lines+=['','B minus controls (AP points):']
        for mode,delta in summary['b_minus_controls_ap_points'].items():lines.append(f'- {names[mode]}: {delta:+.6f}')
        lines+=['','All ten metrics and postprocessing records match the public evaluator.',
            'The original internal-development experiment is retained separately in ../results.',
            'Large caches, gain shards and all epoch checkpoints stay on the experiment drive.','']
        (temp/'README.md').write_text('\n'.join(lines),encoding='utf-8')
        os.replace(temp,destination)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);raise
    print('Copied and verified aligned reports plus best.pt:',destination)
    for _,relative in plan:print(' ',relative)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,default=LABELS)
    p.add_argument('--train',type=Path,default=TRAINING)
    p.add_argument('--evaluate',type=Path,default=EVALUATION)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results_ap')
    args=p.parse_args()
    collect(args.labels.resolve(),args.train.resolve(),args.evaluate.resolve(),args.output.resolve(),
            Path(__file__).resolve().with_name('history.zip'))


if __name__=='__main__':main()
