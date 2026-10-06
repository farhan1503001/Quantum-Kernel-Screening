import os as _os
DATA = _os.environ.get('QKS_DATA_DIR', _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data'))
OUT  = _os.environ.get('QKS_FIG_DIR', _os.path.dirname(_os.path.abspath(__file__)))
"""Figure 3. F1 learning curves for all 30 configurations, N in {50,...,1000} on a fixed test set.
Rows 1-3 main grid (text, vision, multimodal); rows 4-5 the held-out multilingual block."""
import json, numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,'axes.labelweight':'bold',
                     'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.9,'pdf.fonttype':42,'ps.fonttype':42})
S=json.load(open(_os.path.join(DATA, 'grid_split.json')))
def pretty(tag):
    m={'ag_news':'AG News (BERT)','sst2_bert':'SST-2 (BERT)','agnews_minilm':'AG News (MiniLM)','sst2_minilm':'SST-2 (MiniLM)',
       'agnews_roberta':'AG News (RoBERTa)','sst2_roberta':'SST-2 (RoBERTa)','cifar10':'CIFAR-10 (CLIP)','fashion_clip':'Fashion (CLIP)',
       'cifar10_dinov2':'CIFAR-10 (DINOv2)','fashion_dinov2':'Fashion (DINOv2)','cifar10_vit':'CIFAR-10 (ViT)','fashion_vit':'Fashion (ViT)',
       'mscoco':'MSCOCO (CLIP)','snlive':'SNLI-VE (CLIP)','mscoco_siglip':'MSCOCO (SigLIP)','snlive_siglip':'SNLI-VE (SigLIP)',
       'mscoco_flava':'MSCOCO (FLAVA)','snlive_flava':'SNLI-VE (FLAVA)'}
    if tag in m: return m[tag]
    _,enc,lang=tag.split('_'); return f"{lang.capitalize()} ({'mBERT' if enc=='mbert' else 'XLM-R'})"
DOM=['text','vision','multimodal','multilingual']; ENC={'text':['BERT','MiniLM','RoBERTa'],'vision':['CLIP','DINOv2','ViT'],'multimodal':['CLIP','SigLIP','FLAVA'],'multilingual':['mBERT','XLM-R']}
rows=sorted(S['main'],key=lambda r:(DOM.index(r['domain']),ENC[r['domain']].index(r['encoder']),r['tag']))
rows+=sorted(S['heldout'],key=lambda r:(ENC['multilingual'].index(r['encoder']),r['tag']))
ncols=6; nrows=5
fig,axs=plt.subplots(nrows,ncols,figsize=(7.2,7.6),sharex=True)
for i,r in enumerate(rows):
    ax=axs[i//ncols][i%ncols]; lc=r['learning_curve']; x=lc['sizes']
    ax.plot(x,lc['quantum'],'-o',color='#b45309',lw=1.2,ms=2.8,label='Quantum (IQP)')
    ax.plot(x,lc['linear'],'-s',color='#1f7a3d',lw=1.0,ms=2.4,label='Linear SVM')
    ax.plot(x,lc['rbf'],'--^',color='#1f7a3d',lw=1.0,ms=2.6,alpha=0.7,label='RBF SVM')
    ax.set_xscale('log'); ax.set_xticks([50,100,200,400,1000]); ax.set_xticklabels(['50','100','200','400','1000'])
    ax.set_title(pretty(r['tag']),fontsize=7.0,pad=2); ax.tick_params(direction='in',length=2.5,width=0.8)
    ax.grid(color='#e6e6e6',lw=0.5); ax.set_axisbelow(True)
    if i%ncols==0: ax.set_ylabel('F1')
    if i//ncols==nrows-1: ax.set_xlabel('N train')
# row labels on the left margin
fig.text(0.012,0.93,'Text',rotation=90,va='center',ha='center',fontsize=8,fontweight='bold')
fig.text(0.012,0.745,'Vision',rotation=90,va='center',ha='center',fontsize=8,fontweight='bold')
fig.text(0.012,0.56,'Multimodal',rotation=90,va='center',ha='center',fontsize=8,fontweight='bold')
fig.text(0.012,0.28,'Held-out (multilingual)',rotation=90,va='center',ha='center',fontsize=8,fontweight='bold')
# separator between main grid and held-out block
fig.add_artist(plt.Line2D([0.045,0.99],[0.448,0.448],transform=fig.transFigure,color='#888888',lw=0.9,ls='--'))
h,l=axs[0][0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,0.0),fontsize=7.5)
fig.subplots_adjust(left=0.085,right=0.99,bottom=0.085,top=0.96,wspace=0.32,hspace=0.50)
for ext in ('pdf','png','svg'): fig.savefig(f'{OUT}/fig3_curves.{ext}',dpi=300,bbox_inches='tight')
print("panels:",len(rows))
