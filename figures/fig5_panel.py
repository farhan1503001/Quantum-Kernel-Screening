import os as _os
DATA = _os.environ.get('QKS_DATA_DIR', _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data'))
OUT  = _os.environ.get('QKS_FIG_DIR', _os.path.dirname(_os.path.abspath(__file__)))
"""Figure 4, single column. The screening rule fit on the 18-config main grid applied to the
12-config multilingual hold-out. Filled markers are flagged (A_C<0.05)."""
import json, numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9.5,'axes.labelweight':'bold',
                     'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
                     'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42})
S=json.load(open(_os.path.join(DATA, 'grid_split.json'))); main,held=S['main'],S['heldout']
g=np.array([r['gap'] for r in main]); kc=np.array([r['c_kta'] for r in main])
gh=np.array([r['gap'] for r in held]); kh=np.array([r['c_kta'] for r in held]); enc=[r['encoder'] for r in held]
flagged=kh<0.05; MKE={'mBERT':'D','XLM-R':'p'}
LCOL={'hindi':'#0072B2','arabic':'#E69F00','chinese':'#009E73','spanish':'#CC79A7','german':'#D55E00','english':'#56B4E9'}
lang_of=lambda t:t.split('_')[-1]
LANG={r['tag']:r['tag'].split('_')[-1].capitalize() for r in held}
OFF={'multiling_mbert_hindi':(-0.004,-0.0065,'left'),'multiling_mbert_arabic':(0.004,0.0045,'left'),
     'multiling_xlmr_hindi':(0.004,0.004,'left'),'multiling_mbert_english':(-0.004,0.0045,'right'),
     'multiling_mbert_spanish':(0.004,0.0045,'left'),'multiling_xlmr_english':(0.004,0.006,'left'),
     'multiling_mbert_chinese':(0.005,-0.0055,'left'),'multiling_xlmr_arabic':(0.004,0.0045,'left'),
     'multiling_xlmr_spanish':(-0.004,0.006,'right'),'multiling_xlmr_german':(-0.004,-0.006,'right'),
     'multiling_xlmr_chinese':(0.004,-0.0055,'left'),'multiling_mbert_german':(0.004,0.005,'left')}

fig,ax=plt.subplots(figsize=(4.35,2.77)); fig.subplots_adjust(left=0.155,right=0.985,bottom=0.30,top=0.975)
ax.axvspan(0,0.05,color='#d62728',alpha=0.10,lw=0,zorder=0); ax.axvline(0.05,color='#d62728',lw=1.0,ls='--',zorder=1)
ax.axhline(0,color='k',lw=0.6,zorder=1)
b1,a1=np.polyfit(kc,g,1); n=len(kc); resid=g-(a1+b1*kc); s=np.sqrt(np.sum(resid**2)/(n-2)); xm=kc.mean(); Sxx=np.sum((kc-xm)**2)
t=stats.t.ppf(0.975,n-2); xs=np.linspace(0,0.135,80); yh=a1+b1*xs; se=s*np.sqrt(1/n+(xs-xm)**2/Sxx)
ax.fill_between(xs,yh-t*se,yh+t*se,color='#bdbdbd',alpha=0.35,lw=0,zorder=2.5); ax.plot(xs,yh,'-',color='k',lw=1.1,zorder=3)
for k,r in enumerate(held):
    e=r['encoder']; c=LCOL[lang_of(r['tag'])]
    if flagged[k]: ax.scatter(kh[k],gh[k],s=80,marker=MKE[e],color=c,edgecolor='k',lw=0.7,zorder=4)
    else:          ax.scatter(kh[k],gh[k],s=80,marker=MKE[e],facecolor='white',edgecolor=c,lw=1.4,zorder=4)
ax.set_xlim(-0.006,0.135); ax.set_ylim(-0.09,0.045)
ax.set_xlabel('Classical alignment $\\mathbf{A_C}$'); ax.set_ylabel('Actual gap $\\mathbf{\\Delta}$')
ax.tick_params(direction='in',length=4,width=1.2,top=False,right=False)
from matplotlib.lines import Line2D
h=[Line2D([],[],marker='D',color='w',markerfacecolor='#7f7f7f',markeredgecolor='k',markersize=8,label='mBERT'),
   Line2D([],[],marker='p',color='w',markerfacecolor='#7f7f7f',markeredgecolor='k',markersize=9,label='XLM-R')]
h+=[Line2D([],[],marker='o',color='w',markerfacecolor=LCOL[l],markeredgecolor='k',markersize=8,label=l.capitalize()) for l in ['hindi','arabic','chinese','spanish','german','english']]
fig.legend(handles=h,loc='lower center',ncol=4,frameon=False,handletextpad=0.25,columnspacing=1.0,bbox_to_anchor=(0.5,-0.015),fontsize=7.8)
for ext in ('pdf','png'): fig.savefig(f'{OUT}/fig5_panel.{ext}',dpi=300,bbox_inches=None)
print(f"flagged {flagged.sum()}/12 mean {gh[flagged].mean():+.4f} max {gh[flagged].max():+.4f} | unflagged mean {gh[~flagged].mean():+.4f}")
