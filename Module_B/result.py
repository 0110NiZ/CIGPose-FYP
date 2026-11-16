"""Copy completed reports and the best B weights into this folder for GitHub."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Module_B.config import LABELS, TRAINING, EVALUATION, EXPECTED_DETECTIONS
from Module_B.common import sha, digest, atomic_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_reports(labels, training, evaluation, history):
    lr, ls = read_json(labels/'run.json'), read_json(labels/'summary.json')
    tr, ts = read_json(training/'run.json'), read_json(training/'summary.json')
    er, es = read_json(evaluation/'run.json'), read_json(evaluation/'summary.json')
    if (lr.get('scope') != 'full' or ls.get('passed') is not True
            or ls.get('fingerprint') != digest(lr) or ls.get('instances') != 149813
            or ls.get('candidate_records') != 599252):
        raise ValueError('Full label report/provenance mismatch')
    if (tr.get('scope') != 'full' or ts.get('training_completed') is not True
            or ts.get('run_fingerprint') != digest(tr)
            or tr.get('labels_fingerprint') != digest(lr)
            or tr.get('labels_summary_sha256') != sha(labels/'summary.json')
            or ts.get('best_checkpoint_sha256') != sha(training/'best.pt')
            or tr.get('split_sha256') != digest(read_json(training/'split.json'))):
        raise ValueError('Training report/weights/provenance mismatch')
    if (er.get('scope') != 'full' or es.get('scope') != 'full'
            or es.get('formal_coco_ap_evaluated') is not True
            or es.get('prediction_checks_passed') is not True
            or es.get('fingerprint') != digest(er)
            or er.get('b_training_fingerprint') != digest(tr)
            or er.get('b_checkpoint_sha256') != sha(training/'best.pt')
            or er.get('b_training_summary_sha256') != sha(training/'summary.json')
            or es.get('detections') != EXPECTED_DETECTIONS):
        raise ValueError('Evaluation report/model/provenance mismatch')
    methods = {'p0_current','fixed_alpha025','top1_alpha1_current','b_current'}
    if set(es.get('evaluations',{})) != methods or set(es.get('ap_0_to_1',{})) != methods:
        raise ValueError('Expected all four comparison methods')
    for method, result in es['evaluations'].items():
        if (not result['all_10_metrics_match'] or not result['postprocessing_records_match']
                or result['audit']['evaluation_image_count'] != 5000
                or result['input_count'] != EXPECTED_DETECTIONS
                or read_json(evaluation/method/'metrics.json') != result):
            raise ValueError('Scoring consistency or coverage mismatch: '+method)
    # The archive keeps exactly the original bytes referenced by these runs.
    with zipfile.ZipFile(history) as archive:
        import hashlib
        known = {name: hashlib.sha256(archive.read(name)).hexdigest()
                 for name in archive.namelist() if name.endswith('.py')}
    if tr.get('script_sha256') != known['train_b_gain.py']:
        raise ValueError('Original training source does not match history.zip')
    for run, name in ((lr,'generate_b_labels_v4.py'), (er,'evaluate_b_detection.py')):
        values = [value for path,value in run['code_sha256'].items() if Path(path).name == name]
        if values != [known[name]]:
            raise ValueError('Original experiment source does not match history.zip: '+name)
    return es


def collect(labels, training, evaluation, destination, history):
    for source in (labels,training,evaluation):
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError('Keep report snapshot separate from original experiment directories')
    summary = check_reports(labels,training,evaluation,history)
    plan = []
    for source, subfolder, names in (
        (labels,'labels',('run.json','summary.json','resolved_config.py')),
        (training,'train',('run.json','summary.json','split.json','history.json','best.pt')),
        (evaluation,'evaluate',('run.json','summary.json')),
    ):
        for name in names:
            file = source/name
            if not file.is_file():
                raise FileNotFoundError(file)
            plan.append((file,Path(subfolder)/name))
    for method in summary['evaluations']:
        plan.append((evaluation/method/'metrics.json',Path('evaluate')/method/'metrics.json'))
    manifest = dict(files={str(relative):sha(source) for source,relative in plan},
        history_sha256=sha(history),
        original_directories=dict(labels=str(labels),train=str(training),evaluate=str(evaluation)),
        note='Original files copied byte for byte. No report fingerprints or weights changed.')
    if destination.exists():
        if read_json(destination/'files.json') != manifest:
            raise FileExistsError('Different report snapshot exists: '+str(destination))
        for _,relative in plan:
            if sha(destination/relative) != manifest['files'][str(relative)]:
                raise ValueError('Existing copied report changed: '+str(relative))
        print('Reports already copied and verified:',destination)
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.results-',dir=destination.parent))
    try:
        for source,relative in plan:
            target=temp/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            if sha(target) != manifest['files'][str(relative)]:
                raise ValueError('Report copy differs: '+str(relative))
        atomic_json(temp/'files.json',manifest)
        names={'p0_current':'P0','fixed_alpha025':'Fixed alpha=0.25',
               'top1_alpha1_current':'Top-1 alpha=1','b_current':'B'}
        lines=['# Module B results','','COCO val2017, fixed detection boxes, single view.',
            'All four methods use current-runtime predictions and the same scoring protocol.',
            '', '| Method | AP |','| --- | ---: |']
        for method,ap in summary['ap_0_to_1'].items():
            lines.append(f'| {names[method]} | {100*ap:.4f} |')
        lines += ['', 'B improves on Top-1, but does not beat fixed alpha=0.25.',
            'The difference from P0 is small; no statistical significance claim is made.',
            '', 'The reports retain their original filenames, paths, fingerprints and source hashes.',
            'Historical source code is in ../history.zip.',
            'Large feature caches, gain shards, raw predictions and the base CIGPose checkpoint remain outside Git.','']
        (temp/'README.md').write_text('\n'.join(lines),encoding='utf-8')
        os.replace(temp,destination)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    print('Copied and verified reports plus best.pt:',destination)
    for relative in manifest['files']:
        print(' ',relative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,default=LABELS)
    parser.add_argument('--train',type=Path,default=TRAINING)
    parser.add_argument('--evaluate',type=Path,default=EVALUATION)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    collect(args.labels.resolve(),args.train.resolve(),args.evaluate.resolve(),
            args.output.resolve(),Path(__file__).resolve().with_name('history.zip'))


if __name__=='__main__':
    main()
