"""Recreate v11 training curves from history.json; does not touch model weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def plot_history(history_file: str, output: str | None = None) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(history_file)
    hist=json.loads(p.read_text(encoding='utf-8'))
    if not hist:
        raise ValueError('Empty history')
    epochs=np.array([d['epoch'] for d in hist],dtype=int)
    fig,ax=plt.subplots(2,1,figsize=(10,8),constrained_layout=True)
    ax[0].plot(epochs,[d['train_loss'] for d in hist],label='Train total',linewidth=1.8)
    ax[0].plot(epochs,[d['val_loss'] for d in hist],label='Validation total',linewidth=1.8)
    ax[0].set(xlabel='Epoch',ylabel='Total composite loss',title='FNO / GNO training')
    ax[0].legend();ax[0].grid(alpha=.25)
    parts=['data','wet_support','velocity','momentum_consistency','admissible','admissible_l1']
    for key in parts:
        vals=np.array([d.get('val_parts',{}).get(key,np.nan) for d in hist],dtype=float)
        if np.isfinite(vals).any():
            ax[1].plot(epochs,vals,label=key,linewidth=1.3)
    ax[1].set(xlabel='Epoch',ylabel='Validation loss component (unweighted)',title='Individual validation components')
    ax[1].legend(loc='best',fontsize=8,ncol=2);ax[1].grid(alpha=.25)
    out=Path(output) if output else p.with_name('training_curves_v11.png')
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=170,bbox_inches='tight')
    plt.close(fig)
    return str(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--history',required=True)
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    print(plot_history(args.history,args.out))
