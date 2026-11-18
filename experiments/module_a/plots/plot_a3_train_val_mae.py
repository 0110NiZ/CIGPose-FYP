"""Comparable end-of-epoch train/validation gain MAE; reads the unified results.html for data and provenance."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
payload=(ROOT.parent/'results.html').read_text(encoding='utf-8').split('<script id="experiment-data" type="application/json">',1)[1].split('</script>',1)[0]
record=json.loads(payload)['experiments']['module_a_a3_full_joint_context_seed42_bs1024_eval1']
data=dict(record['regression_diagnostic'],history=record['history'])
assert data['complete'] and len(data['history'])==20
# Use a locally installed Chinese font; no downloads or machine-specific paths.
import os
custom=os.environ.get('CIG_PLOT_FONT')
if custom:
    font_manager.fontManager.addfont(custom)
    family=font_manager.FontProperties(fname=custom).get_name()
else:
    candidates=['Microsoft YaHei','Noto Sans CJK SC','Noto Sans SC','SimHei','WenQuanYi Zen Hei']
    installed={f.name for f in font_manager.fontManager.ttflist}
    family=next((f for f in candidates if f in installed),None)
    if family is None:
        raise RuntimeError('Install a Chinese font or set CIG_PLOT_FONT to its font file.')
plt.rcParams.update({'font.family':family,'axes.unicode_minus':False,'font.size':14,'pdf.fonttype':42,'svg.fonttype':'none'})
h=data['history'];epochs=[r['epoch'] for r in h]
fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
ax.plot(epochs,[r['train_eval_gain_mae'] for r in h],label='训练集',color='#254D75',lw=2.4,marker='o',ms=4.5)
ax.plot(epochs,[r['val_eval_gain_mae'] for r in h],label='验证集',color='#DF8035',lw=2.4,marker='s',ms=4.5)
ax.set_xlabel('训练轮次',labelpad=12)
ax.set_ylabel('收益预测误差（MAE）',labelpad=12)
ax.set_xticks([1,2,4,6,8,10,12,14,16,18,20]);ax.set_xlim(.5,20.5)
ax.ticklabel_format(axis='y',style='plain',useOffset=False)
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#ADB8C4')
ax.grid(axis='y',color='#E5EAF0',lw=.8);ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=14)
for ext in ('png','pdf','svg'):fig.savefig(ROOT/('A3_train_val_MAE.'+ext),dpi=240,facecolor='white')

print(json.dumps({'first':h[0],'last':h[-1],'min_val_epoch':min(h,key=lambda r:r['val_eval_gain_mae'])['epoch']},indent=2))
