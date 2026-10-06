import os as _os
DATA = _os.environ.get('QKS_DATA_DIR', _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data'))
OUT  = _os.environ.get('QKS_FIG_DIR', _os.path.dirname(_os.path.abspath(__file__)))
"""Figure 3, journal-conventional style with matched inset zooms. Main grid, 18 configurations."""
import json, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9.5,'axes.labelweight':'bold',
                     'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
                     'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42})

canon = json.load(open(_os.path.join(DATA, 'grid_split.json')))['main']
gap=np.array([c['gap'] for c in canon]); kq=np.array([c['kta'] for c in canon]); kc=np.array([c['c_kta'] for c in canon])
doms=[c['domain'] for c in canon]
C={'text':'#0072B2','vision':'#E69F00','multimodal':'#009E73'}
LBL={'text':'Text','vision':'Vision','multimodal':'Multimodal'}; MK={'text':'o','vision':'s','multimodal':'^'}
INSET_BOX=(0.30, 0.58, 0.38, 0.40)          # identical inset geometry in both panels
ZX=(-0.004, 0.066)                          # zoomed x-range in both panels

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 3.1))
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.30, top=0.90, wspace=0.42)
from sklearn.linear_model import LinearRegression

def pts(ax, x, y, s=62, lw=0.7, label=False):
    for d in ['text','vision','multimodal']:
        m=[i for i,dd in enumerate(doms) if dd==d]
        ax.scatter(x[m], y[m], s=s, marker=MK[d], color=C[d], edgecolor='k', lw=lw, label=LBL[d] if label else None, zorder=4)

def fit(ax, x, y, xr, band=True):
    b1,a1=np.polyfit(x,y,1); n=len(x); resid=y-(a1+b1*x); s=np.sqrt(np.sum(resid**2)/(n-2))
    xm=x.mean(); Sxx=np.sum((x-xm)**2); t=stats.t.ppf(0.975,n-2)
    xs=np.linspace(*xr,120); yh=a1+b1*xs; se=s*np.sqrt(1/n+(xs-xm)**2/Sxx)
    if band: ax.fill_between(xs, yh-t*se, yh+t*se, color='#bdbdbd', alpha=0.35, lw=0, zorder=2.5)
    ax.plot(xs, yh, '-', color='k', lw=1.1, zorder=3)

def inset(ax, x, y, yr, yticks):
    ins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=INSET_BOX, bbox_transform=ax.transAxes, loc='lower left', borderpad=0)
    pts(ins, x, y, s=34, lw=0.5)
    ins.set_xlim(*ZX); ins.set_ylim(*yr); ins.set_xticks([0, 0.05]); ins.set_yticks(yticks)
    ins.tick_params(labelsize=5.8, direction='in', length=2.5, width=0.8, pad=2)
    for s in ins.spines.values(): s.set_linewidth(0.9)
    mark_inset(ax, ins, loc1=3, loc2=4, fc='none', ec='#555555', lw=0.7)
    return ins

# ---------------- (a) ----------------
axA.axvspan(0, 0.05, color='#d62728', alpha=0.10, lw=0, zorder=0)
axA.axvline(0.05, color='#d62728', lw=1.0, ls='--', zorder=1)
axA.axhline(0, color='k', lw=0.6, zorder=1)
pts(axA, kc, gap, label=True)
fit(axA, kc, gap, (0, kc.max()*1.02))
axA.set_xlim(-0.02, kc.max()*1.06); axA.set_ylim(-0.078, 0.062)
axA.text(0.96, 0.05, '$r = 0.873$, $p = 2.2\\times10^{-6}$\nLOO-CV $R^2 = 0.709$', transform=axA.transAxes, ha='right', va='bottom',
         fontsize=7.4, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k', lw=0.6))
axA.set_xlabel('Classical kernel-target alignment $\\mathbf{A_C}$')
axA.set_ylabel('Utility gap $\\mathbf{\\Delta = F1_Q - F1_C}$')

axA.text(0.5, 1.02, '(a)', transform=axA.transAxes, fontsize=10, fontweight='bold', ha='center', va='bottom')
iA = inset(axA, kc, gap, (-0.070, 0.006), [-0.06, -0.03, 0])
iA.axhline(0, color='k', lw=0.5); iA.axvspan(0, 0.05, color='#d62728', alpha=0.10, lw=0, zorder=0)
iA.axvline(0.05, color='#d62728', lw=0.8, ls='--')
iA.text(0.965, 0.50, 'screening region', transform=iA.transAxes, ha='right', va='center', rotation=90, fontsize=5.6, color='#b71c1c')

# ---------------- (b) ----------------
lim=max(kq.max(),kc.max())*1.08
axB.plot([0,lim],[0,lim], color='#9e9e9e', lw=0.8, ls=':', zorder=1)
pts(axB, kc, kq)
fit(axB, kc, kq, (0, lim), band=False)
axB.set_xlim(-0.02, lim); axB.set_ylim(-0.02, lim)
axB.text(0.96, 0.05, '$r = 0.951$', transform=axB.transAxes, ha='right', va='bottom', fontsize=7.4,
         bbox=dict(boxstyle='square,pad=0.35', fc='white', ec='k', lw=0.6))
axB.set_xlabel('Classical alignment $\\mathbf{A_C}$'); axB.set_ylabel('Quantum alignment $\\mathbf{A_Q}$')
axB.text(0.5, 1.02, '(b)', transform=axB.transAxes, fontsize=10, fontweight='bold', ha='center', va='bottom')
iB = inset(axB, kc, kq, (-0.006, 0.075), [0, 0.03, 0.06])


# ---------------- (c) leave-one-domain-out ----------------
pred=np.zeros(len(gap))
for d in ['text','vision','multimodal']:
    tr=[i for i in range(len(gap)) if doms[i]!=d]; te=[i for i in range(len(gap)) if doms[i]==d]
    pred[te]=LinearRegression().fit(kc[tr].reshape(-1,1),gap[tr]).predict(kc[te].reshape(-1,1))
lo_=min(pred.min(),gap.min())-0.012; hi_=max(pred.max(),gap.max())+0.012
axC.plot([lo_,hi_],[lo_,hi_],color='#9e9e9e',lw=0.8,ls=':',zorder=1)
axC.axhline(0,color='k',lw=0.6,zorder=1); axC.axvline(0,color='k',lw=0.6,zorder=1)
pts(axC, pred, gap)
rl=stats.pearsonr(pred,gap); mae=np.abs(pred-gap).mean()
axC.text(0.96,0.05,f'$r = {rl[0]:.3f}$\nMAE $= {mae:.3f}$',transform=axC.transAxes,ha='right',va='bottom',fontsize=7.4,
         bbox=dict(boxstyle='square,pad=0.3',fc='white',ec='k',lw=0.6))
axC.set_xlim(lo_,hi_); axC.set_ylim(lo_,hi_)
axC.set_xlabel('Predicted gap $\\mathbf{\\hat{\\Delta}}$'); axC.set_ylabel('Actual gap $\\mathbf{\\Delta}$')
axC.text(0.5,1.02,'(c)',transform=axC.transAxes,fontsize=10,fontweight='bold',ha='center',va='bottom')

for ax in (axA, axB, axC): ax.tick_params(direction='in', length=4, width=1.2, top=False, right=False)
handles, labels = axA.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, handletextpad=0.3, columnspacing=2.0,
           bbox_to_anchor=(0.5, 0.0), markerscale=1.1)
for ext in ('pdf','png'):
    fig.savefig(f'{OUT}/fig4_screen.{ext}', dpi=300, bbox_inches='tight')
r=stats.pearsonr(kc,gap); ra=stats.pearsonr(kc,kq)
print(f"n={len(kc)} r_AC={r[0]:.3f} p={r[1]:.1e} agreement r={ra[0]:.3f}")
