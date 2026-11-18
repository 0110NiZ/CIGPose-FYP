"""One self-contained result file: human-readable tables plus embedded records."""
import argparse
import fcntl
import html
import json
from pathlib import Path
import config

RESULT=config.WORK/'code/experiments/module_a/results.html'
START='<script id="experiment-data" type="application/json">'
END='</script>'

def read_result(path=RESULT):
    if not path.exists():return dict(experiments={},baselines={},notes=[])
    text=path.read_text(encoding='utf-8')
    return json.loads(text.split(START,1)[1].split(END,1)[0])

def number(value,digits=4):
    return '—' if value is None else format(value,'.%df'%digits)

def write_result(data,path=RESULT):
    path.parent.mkdir(parents=True,exist_ok=True)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>模块A实验结果</title>',
        '<style>body{font:16px/1.6 system-ui;margin:40px auto;max-width:1200px;padding:0 20px;color:#172334}table{border-collapse:collapse;width:100%;margin:16px 0 30px}td,th{padding:8px 12px;border:1px solid #d9dfe7;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf1f6}.best{background:#e5f5eb;font-weight:600}small{color:#596579}h2{margin-top:36px}</style>',
        '<h1>模块 A 实验结果</h1><p>AP/AR 均为 0–100 标度，越高越好。最终模型取完整验证 AP 最高轮；并列取最早轮。“—”表示该项未记录或当轮未验证，不代表零。</p>']
    for note in data.get('notes',[]):parts.append('<p>'+html.escape(note)+'</p>')
    parts.append('<h2>最终结果</h2><table><tr><th>实验 / 基线</th><th>最佳轮次</th><th>AP</th><th>AR</th></tr>')
    for name,rec in data.get('baselines',{}).items():
        m=rec.get('metrics',rec)
        parts.append('<tr><td>%s</td><td>不训练</td><td>%s</td><td>%s</td></tr>'%(html.escape(name),number(m['coco/AP']*100),number(m['coco/AR']*100)))
    for name,rec in data['experiments'].items():
        valid=[r for r in rec['history'] if r.get('validation')]
        if not valid:continue
        best=max(valid,key=lambda r:r['validation']['coco/AP']);m=best['validation']
        parts.append('<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>'%(html.escape(rec.get('title',name)),best['epoch'],number(m['coco/AP']*100),number(m['coco/AR']*100)))
    parts.append('</table>')
    for name,rec in data['experiments'].items():
        valid=[r for r in rec['history'] if r.get('validation')]
        best=max(valid,key=lambda r:r['validation']['coco/AP'])['epoch'] if valid else None
        parts.append('<h2>%s</h2><p>%s</p><small>保留模型：%s</small>'%(html.escape(rec.get('title',name)),html.escape(rec.get('description','')),html.escape(rec.get('checkpoint',''))))
        parts.append('<table><tr><th>轮次</th><th>训练Loss</th><th>训练选中收益</th><th>AP</th><th>AP50</th><th>AP75</th><th>AR</th><th>标记</th></tr>')
        for row in rec['history']:
            m=row.get('validation') or {};gain=row.get('train_mean_gain',row.get('selection',{}).get('mean_gain'))
            values=[number(row.get('loss'),6),number(gain,8)]+[number(m[k]*100) if k in m else '—' for k in ['coco/AP','coco/AP .5','coco/AP .75','coco/AR']]
            parts.append('<tr%s><td>%d</td>%s<td>%s</td></tr>'%(' class="best"' if row['epoch']==best else '',row['epoch'],''.join('<td>'+v+'</td>' for v in values),'最终最佳' if row['epoch']==best else ('未验证' if not m else '')))
        parts.append('</table>')
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    parts.append(START+payload+END+'</html>')
    temporary=path.with_suffix('.tmp');temporary.write_text('\n'.join(parts),encoding='utf-8');temporary.replace(path)

def update_experiment(name,record):
    with open('/tmp/cig-module-a-result.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=read_result();data['experiments'][name]=record;write_result(data)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args()
    print(RESULT)
    for name,rec in read_result()['experiments'].items():
        rows=[r for r in rec['history'] if r.get('validation')]
        if rows:
            best=max(rows,key=lambda r:r['validation']['coco/AP'])
            print('%s: best epoch %d, AP %.6f'%(name,best['epoch'],best['validation']['coco/AP']*100))
