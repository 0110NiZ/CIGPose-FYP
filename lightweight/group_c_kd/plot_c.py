import csv
import json
from pathlib import Path

def plot(work, experiment='C (SimCC KD)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work=Path(work)
    rows=list(csv.DictReader((work/'loss_history.csv').open()))
    ap=json.loads((work/'ap_history.json').read_text()) if (work/'ap_history.json').exists() else []
    x=[int(r['epoch']) for r in rows]
    train=[float(r['train_loss_kpt_per_person']) for r in rows]
    val=[float(r['val_loss_kpt_per_person']) for r in rows]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,start,title in [(axes[0],1,'Train vs validation loss'),(axes[1],70,'Late-stage loss (epochs 70–100)')]:
        ax.plot(x,train,label='Train: online, augmented',color='#2563eb',lw=2,marker='.',ms=3)
        ax.plot(x,val,label='Validation: eval, no augmentation',color='#ea580c',lw=2,marker='.',ms=3)
        ax.set_xlim(start,max(100,max(x)));ax.set_xlabel('Epoch');ax.set_ylabel('Keypoint loss per person')
        ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.axvspan(93.5,100,alpha=.1,color='orange');ax.axvline(93.5,ls=':',color='orange')
        if start==70:
            late=[y for e,t,v in zip(x,train,val) if e>=70 for y in (t,v)]
            if late:
                lo,hi=min(late),max(late);pad=max((hi-lo)*.1,1e-5);ax.set_ylim(lo-pad,hi+pad)
            else:
                ax.text(.5,.5,'Available after epoch 70',ha='center',transform=ax.transAxes)
    axes[2].plot([r['epoch'] for r in ap],[r['coco/AP']*100 for r in ap],'o-',color='#059669')
    axes[2].set_title('Full validation AP (EMA + flip test)')
    axes[2].set_xlabel('Epoch');axes[2].set_ylabel('AP (0–100)');axes[2].grid(alpha=.2)
    if not ap:
        axes[2].text(.5,.5,'First AP evaluation: epoch 10',ha='center',transform=axes[2].transAxes)
    fig.suptitle(f'CIGPose {experiment}: batch128 | 30% fixed subset | FP16 | 256-d Head',fontsize=15)
    fig.text(.05,.025,'Both loss curves show supervised keypoint KL summed over all batches and divided by the number of people.\nValidation loss uses raw weights, FP16 and a single view. AP uses FP32 EMA and flip testing every 10 epochs. Shading marks stage-2 augmentation.',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.91))
    for ext in ('png','pdf','svg'):fig.savefig(work/('train_val_loss.'+ext),dpi=180)
    plt.close(fig)

if __name__=='__main__':
    import sys
    plot(Path(sys.argv[1]))
