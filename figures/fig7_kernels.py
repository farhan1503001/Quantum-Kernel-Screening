import os as _os
DATA = _os.environ.get('QKS_DATA_DIR', _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data'))
OUT  = _os.environ.get('QKS_FIG_DIR', _os.path.dirname(_os.path.abspath(__file__)))
"""Figure 6. The screen under six quantum kernels. Gap against classical RBF alignment A_C (log axis),
all 30 configurations from featuremap_ablation_all30.json. Main grid by domain; hold-out as open diamonds."""
import json, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,'axes.labelweight':'bold',
                     'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42})
R=json.load(open(_os.path.join(DATA, 'featuremap_ablation_all30.json')))
S=json.load(open(_os.path.join(DATA, 'grid_split.json')))
main=[r['tag'] for r in S['main']]; held=[r['tag'] for r in S['heldout']]
dom={r['tag']:r['domain'] for r in S['main']}
C={'text':'#0072B2','vision':'#E69F00','multimodal':'#009E73'}; MK={'text':'o','vision':'s','multimodal':'^'}; HC='#CC79A7'
PANELS=[('IQP-1L','(a) IQP, 1 layer, 8 qubits'),('IQP-2L','(b) IQP, 2 layers, 8 qubits'),('Angle','(c) Angle encoding, 8 qubits'),
        ('IQP-1L-all','(d) IQP, all-pairs ZZ, 8 qubits'),('IQP-1L-6q','(e) IQP, 1 layer, 6 qubits'),('IQP-1L-4q','(f) IQP, 1 layer, 4 qubits')]
allg=[R[t]['kernels'][k]['gap'] for t in R for k,_ in PANELS]; YL=(min(allg)-0.012, max(allg)+0.012)
fig,axs=plt.subplots(2,3,figsize=(7.2,5.0),sharex=True,sharey=True)
fig.subplots_adjust(left=0.085,right=0.985,bottom=0.17,top=0.93,wspace=0.10,hspace=0.30)
for ax,(k,title) in zip(axs.flat,PANELS):
    ax.axvspan(0.001,0.05,color='#d62728',alpha=0.10,lw=0,zorder=0); ax.axvline(0.05,color='#d62728',lw=1.0,ls='--',zorder=1)
    ax.axhline(0,color='k',lw=0.6,zorder=1)
    for t in held: ax.scatter(R[t]['c_kta'],R[t]['kernels'][k]['gap'],s=40,marker='D',facecolor='white',edgecolor=HC,lw=1.2,zorder=3)
    for t in main: ax.scatter(R[t]['c_kta'],R[t]['kernels'][k]['gap'],s=52,marker=MK[dom[t]],color=C[dom[t]],edgecolor='k',lw=0.6,zorder=4)
    a=np.array([R[t]['c_kta'] for t in main]); g=np.array([R[t]['kernels'][k]['gap'] for t in main]); r=stats.pearsonr(a,g)[0]
    ax.text(0.97,0.05,f'$r = {r:.3f}$',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,
            bbox=dict(boxstyle='square,pad=0.3',fc='white',ec='k',lw=0.6),zorder=6)
    ax.set_xscale('log'); ax.set_xlim(0.0018,1.1); ax.set_ylim(*YL)
    ax.set_title(title,fontsize=8.6,fontweight='bold',pad=4)
    ax.tick_params(direction='in',length=3.5,width=1.1,top=False,right=False,which='both')
for ax in axs[1]: ax.set_xlabel('Classical alignment $\\mathbf{A_C}$')
for ax in axs[:,0]: ax.set_ylabel('Utility gap $\\mathbf{\\Delta}$')
h=[Line2D([],[],marker='o',color='w',markerfacecolor=C['text'],markeredgecolor='k',markersize=7,label='Text'),
   Line2D([],[],marker='s',color='w',markerfacecolor=C['vision'],markeredgecolor='k',markersize=7,label='Vision'),
   Line2D([],[],marker='^',color='w',markerfacecolor=C['multimodal'],markeredgecolor='k',markersize=7.5,label='Multimodal'),
   Line2D([],[],marker='D',color='w',markerfacecolor='white',markeredgecolor=HC,markeredgewidth=1.2,markersize=6.5,label='Hold-out (multilingual)')]
fig.legend(handles=h,loc='lower center',ncol=4,frameon=False,handletextpad=0.3,columnspacing=1.8,bbox_to_anchor=(0.5,0.0))
for ext in ('pdf','png','svg'): fig.savefig(f'{OUT}/fig7_kernels.{ext}',dpi=300,bbox_inches='tight')
print("r by panel (main grid):",{k:round(stats.pearsonr([R[t]['c_kta'] for t in main],[R[t]['kernels'][k]['gap'] for t in main])[0],3) for k,_ in PANELS})
