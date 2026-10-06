import os as _os
DATA = _os.environ.get('QKS_DATA_DIR', _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data'))
OUT  = _os.environ.get('QKS_FIG_DIR', _os.path.dirname(_os.path.abspath(__file__)))
"""Figure 5, established diagnostics and mechanism, 18-config main grid, four panels."""
import json, numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9.5,'axes.labelweight':'bold',
                     'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
                     'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42})
main=json.load(open(_os.path.join(DATA, 'grid_split.json')))['main']
g=np.array([r['gap'] for r in main]); co=np.array([r['concentration'] for r in main]); gx=np.array([r['g_exact'] for r in main])
oq=np.array([r['offdiag_quantum'] for r in main]); oc=np.array([r['offdiag_rbf'] for r in main]); doms=[r['domain'] for r in main]
lams=[1e-4,1e-5,1e-6,1e-7]; sweep=np.array([[r['g_lambda_sweep'][f"{l:.0e}"] for l in lams] for r in main])
C={'text':'#0072B2','vision':'#E69F00','multimodal':'#009E73'}; MK={'text':'o','vision':'s','multimodal':'^'}
LBL={'text':'Text','vision':'Vision','multimodal':'Multimodal'}
def box(ax,txt,loc='lr'):
    x,ha,y,va=(0.96,'right',0.05,'bottom') if loc=='lr' else (0.04,'left',0.95,'top')
    ax.text(x,y,txt,transform=ax.transAxes,ha=ha,va=va,fontsize=8.5,bbox=dict(boxstyle='square,pad=0.35',fc='white',ec='k',lw=0.6))
def style(ax,letter):
    ax.tick_params(direction='in',length=4,width=1.2,top=False,right=False)
    ax.text(0.5,1.02,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',ha='center',va='bottom')
def scatter(ax,x,y,label=False):
    for d in ['text','vision','multimodal']:
        i=[k for k,dd in enumerate(doms) if dd==d]
        ax.scatter(x[i],y[i],s=95,marker=MK[d],color=C[d],edgecolor='k',lw=0.7,label=LBL[d] if label else None,zorder=4)
def fitline(ax,x,y):
    b1,a1=np.polyfit(x,y,1); n=len(x); resid=y-(a1+b1*x); s=np.sqrt(np.sum(resid**2)/(n-2)); xm=x.mean(); Sxx=np.sum((x-xm)**2)
    t=stats.t.ppf(0.975,n-2); pad=(x.max()-x.min())*0.05; xs=np.linspace(x.min()-pad,x.max()+pad,80); yh=a1+b1*xs; se=s*np.sqrt(1/n+(xs-xm)**2/Sxx)
    ax.fill_between(xs,yh-t*se,yh+t*se,color='#bdbdbd',alpha=0.35,lw=0,zorder=2.5); ax.plot(xs,yh,'-',color='k',lw=1.1,zorder=3)
    ax.set_xlim(x.min()-pad,x.max()+pad)

fig,axs=plt.subplots(2,2,figsize=(7.2,6.6)); (axA,axB),(axC,axD)=axs
fig.subplots_adjust(left=0.10,right=0.985,bottom=0.14,top=0.95,wspace=0.34,hspace=0.42)

# (a) concentration vs gap
axA.axhline(0,color='k',lw=0.6,zorder=1); scatter(axA,co,g,label=True); fitline(axA,co,g)
ra=stats.pearsonr(co,g); box(axA,f'$r = {ra[0]:.3f}$, $p = {ra[1]:.3f}$')
axA.set_xlabel('Spectral concentration $\\mathbf{\\rho}$'); axA.set_ylabel('Utility gap $\\mathbf{\\Delta}$'); axA.set_ylim(-0.078,0.03); style(axA,'(a)')
# (b) exact g vs gap
axB.axhline(0,color='k',lw=0.6,zorder=1); scatter(axB,gx,g); fitline(axB,gx,g)
rb=stats.pearsonr(gx,g); box(axB,f'$r = {rb[0]:.3f}$, $p = {rb[1]:.3f}$')
axB.set_xlabel('Geometric difference $\\mathbf{g}$ ($\\lambda = 10^{-6}$)'); axB.set_ylabel('Utility gap $\\mathbf{\\Delta}$'); axB.set_ylim(-0.078,0.03); style(axB,'(b)')
# (c) g across lambda per configuration
for k,r in enumerate(main):
    axC.plot(lams,sweep[k],'-',color=C[doms[k]],lw=1.0,alpha=0.85,zorder=3)
    axC.scatter(lams,sweep[k],s=28,marker=MK[doms[k]],color=C[doms[k]],edgecolor='k',lw=0.5,zorder=4)
axC.axhline(np.sqrt(1000),color='#555555',lw=0.9,ls='--',zorder=2); axC.text(1.3e-7,np.sqrt(1000)*1.12,'$\\sqrt{N}$',fontsize=8,color='#333333',ha='left',va='bottom')
axC.set_xscale('log'); axC.set_yscale('log'); axC.invert_xaxis(); axC.set_xticks(lams)
axC.set_xlabel('Ridge parameter $\\mathbf{\\lambda}$'); axC.set_ylabel('Geometric difference $\\mathbf{g}$'); style(axC,'(c)')
rs=[stats.pearsonr(sweep[:,j],g)[0] for j in range(4)]
box(axC,f'$r(g,\\Delta)$: {min(rs):.2f} to {max(rs):.2f}\nn.s. at every $\\lambda$',loc='ul')
# (d) mechanism as grouped bars, domain-colored: quantum solid, RBF light with hatch
from matplotlib.patches import Patch
order=['text','vision','multimodal']; w=0.36; xs=np.arange(3)
for i,d in enumerate(order):
    idx=[k for k,dd in enumerate(doms) if dd==d]
    for j,(arr,alpha,hatch) in enumerate([(oq,1.0,None),(oc,0.35,'//')]):
        v=arr[idx]; m=v.mean()
        axD.bar(xs[i]+(j-0.5)*w,m,w,color=C[d],alpha=alpha,hatch=hatch,edgecolor='k',lw=0.8,
                yerr=[[m-v.min()],[v.max()-m]],capsize=3,error_kw=dict(lw=0.9,capthick=0.9,ecolor='k'),zorder=3)
axD.set_xticks(xs); axD.set_xticklabels([LBL[d] for d in order]); axD.set_ylim(0,0.95)
axD.set_ylabel('Mean off-diagonal $\\mathbf{K_{ij}}$'); axD.set_xlabel('Domain')
axD.text(0.96,0.95,f'{oc.mean()/oq.mean():.0f}$\\times$ more concentrated',transform=axD.transAxes,ha='right',va='top',fontsize=8.5,
         bbox=dict(boxstyle='square,pad=0.35',fc='white',ec='k',lw=0.6))
style(axD,'(d)')

h,l=axA.get_legend_handles_labels()
h+= [Patch(facecolor='#7f7f7f',edgecolor='k',label='Quantum kernel'),
     Patch(facecolor='#7f7f7f',alpha=0.35,hatch='//',edgecolor='k',label='RBF kernel')]
l+= ['Quantum kernel','RBF kernel']
fig.legend(h,l,loc='lower center',ncol=5,frameon=False,handletextpad=0.3,columnspacing=1.6,bbox_to_anchor=(0.5,0.01),markerscale=1.1)
for ext in ('pdf','png'): fig.savefig(f'{OUT}/fig6_diagnostics.{ext}',dpi=300,bbox_inches='tight')
print(f"(a) r={ra[0]:+.3f} p={ra[1]:.3f} | (b) r={rb[0]:+.3f} p={rb[1]:.3f} | (c) r range {min(rs):+.3f}..{max(rs):+.3f} | (d) q {oq.mean():.3f} rbf {oc.mean():.3f}")
