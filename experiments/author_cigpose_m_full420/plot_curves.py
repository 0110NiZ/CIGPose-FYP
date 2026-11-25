from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parent
loss=list(csv.DictReader((D/'train_loss_logged_window_means.csv').open()))
ap=json.loads((D/'ap_history.json').read_text())
best=max(ap,key=lambda r:r['coco/AP'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
axes[0].plot([int(r['epoch']) for r in loss],[float(r['loss_kpt']) for r in loss],color='#2563eb',lw=1.7)
axes[0].set_title('Training keypoint loss')
axes[0].set_ylabel('Mean of logged loss windows')
axes[0].axvspan(390.5,420,color='#f59e0b',alpha=.12,label='Stage-2 augmentation')
axes[0].legend(fontsize=8)
for key,label,color in [('coco/AP','AP','#059669'),('coco/AP .5','AP50','#2563eb'),('coco/AP .75','AP75','#ea580c')]:
    axes[1].plot([r['epoch'] for r in ap],[r[key]*100 for r in ap],label=label,color=color,lw=1.8)
axes[1].scatter([best['epoch']],[best['coco/AP']*100],color='#059669',zorder=5,s=35)
axes[1].annotate(f"Best AP {best['coco/AP']*100:.2f} at epoch {best['epoch']}",
                 (best['epoch'],best['coco/AP']*100),xytext=(-155,-30),textcoords='offset points',fontsize=9)
axes[1].set_title('Full val2017: EMA + flip test',pad=35)
axes[1].set_ylabel('AP (0-100)')
axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.set_xlim(1,420)
    ax.grid(alpha=.2)
fig.suptitle('Original CIGPose-m | Full COCO | Batch 384 | FP32 | 420 epochs',fontsize=14)
fig.text(.065,.025,'Training loss is a mean of recorded logger windows. Evaluation uses GT boxes. Validation loss was not recorded.',fontsize=9)
fig.tight_layout(rect=(0,.08,1,.93))
for suffix in ['png','pdf','svg']:
    fig.savefig(D/('training_curves.'+suffix),dpi=180)
plt.close(fig)
print('Full420 curves saved.')
