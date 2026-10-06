import os as _os
OUT = _os.environ.get('QKS_FIG_DIR', _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'
import matplotlib.font_manager as _fm, glob as _glob
for _p in _glob.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-*.otf'):
    if 'cn' not in _p: _fm.fontManager.addfont(_p)
plt.rcParams['font.family']='TeX Gyre Heros'
plt.rcParams['font.weight']='normal'
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Ellipse, Rectangle

np.random.seed(7)
W, H = 12.6, 4.9
Y0=0.25
fig = plt.figure(figsize=(W, H-Y0))
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(Y0,H); ax.axis('off')

DARK='#000000'; BRN='#b45309'; BRN_BG='#fbeed3'; GRN='#1f7a3d'
C_A='#5e3c99'; C_B='#1b9e77'

FS=1.1   # global font scale
def txt(x,y,s,fs=9.5,c=DARK,w='normal',ha='center',va='center',st='normal',z=5):
    ax.text(x,y,s,fontsize=fs*FS,color=c,fontweight=w,ha=ha,va=va,style=st,zorder=z)
def arr(x1,y1,x2,y2,c=DARK,lw=1.7,ms=13,z=3):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=ms,
                                 color=c,linewidth=lw,zorder=z))
def panel(x,y,s):
    ax.add_patch(Circle((x+0.14,y),0.15,facecolor=DARK,zorder=6))
    dy = -0.018 if s in ('b','d') else 0.0
    ax.text(x+0.14,y+dy,s,fontsize=11.5*FS,color='white',fontweight='bold',ha='center',va='center',zorder=7)

def sphere(cx,cy,r,lw=1.4):
    ax.add_patch(Circle((cx,cy),r,facecolor='#fbfbfd',edgecolor=DARK,lw=lw,zorder=2))
    ax.add_patch(Ellipse((cx,cy),2*r,0.62*r,facecolor='none',edgecolor='#8a8a8a',
                         lw=0.9,ls=(0,(3,2)),zorder=3))

def statevec(cx,cy,r,ang_deg,color,lw=2.0,frac=0.97):
    a=np.deg2rad(ang_deg)
    arr(cx,cy,cx+frac*r*np.cos(a),cy+frac*r*np.sin(a),c=color,lw=lw,ms=10)

# ═════════ PANEL (a) ═════════  [compacted leftward]
panel(1.30,4.36,'a'); txt(1.63,4.36,'Feature map',fs=11,w='bold',ha='left')

ax.text(0.18,2.68,r'features $x \in [0,\pi]^{8}$',fontsize=9.5*FS,color='#000000',rotation=90,ha='center',va='center',zorder=5)
ptsA=np.random.randn(9,2)*[0.16,0.20]+[0.72,3.15]
ptsB=np.random.randn(9,2)*[0.16,0.20]+[1.12,2.20]
ax.scatter(ptsA[:,0],ptsA[:,1],s=26,color=C_A,edgecolor='white',lw=0.6,zorder=4)
ax.scatter(ptsB[:,0],ptsB[:,1],s=26,color=C_B,edgecolor='white',lw=0.6,zorder=4)
ax.scatter([ptsA[3,0]],[ptsA[3,1]],s=64,facecolor=C_A,edgecolor=DARK,lw=1.3,zorder=5)
ax.scatter([ptsB[4,0]],[ptsB[4,1]],s=64,facecolor=C_B,edgecolor=DARK,lw=1.3,zorder=5)
txt(ptsA[3,0]-0.20,ptsA[3,1]+0.14,r'$x$',fs=10,w='bold')
txt(ptsB[4,0]+0.22,ptsB[4,1]-0.10,r'$x^{\prime}$',fs=10,w='bold')

arr(1.52,2.75,2.02,2.75,c=BRN,lw=2.0)
txt(1.77,3.00,r'$U(\cdot)$',fs=10,c=BRN,w='bold')

scx,scy,sr=2.92,2.62,0.86
sphere(scx,scy,sr)
statevec(scx,scy,sr,58,C_A)
statevec(scx,scy,sr,96,C_B)
ax.add_patch(Arc((scx,scy),1.05*sr,1.05*sr,theta1=58,theta2=96,edgecolor=DARK,lw=1.1,zorder=4))
txt(scx+0.86*sr*np.cos(np.deg2rad(50)),scy+0.98*sr*np.sin(np.deg2rad(50))+0.26,
    r'$|\phi(x)\rangle$',fs=9.5,c=C_A,w='bold')
txt(scx+0.95*sr*np.cos(np.deg2rad(112)),scy+1.02*sr*np.sin(np.deg2rad(112))+0.10,
    r'$|\phi(x^{\prime})\rangle$',fs=9.5,c=C_B,w='bold')
txt(scx,scy-sr-0.22,r'unit vectors in $\mathbb{C}^{2^{8}}$',fs=9.5,c='#000000',w='normal')
txt(2.30,0.95,r'$K_Q(x,x^{\prime})=\left|\langle\phi(x)|\phi(x^{\prime})\rangle\right|^{2}\in[0,1]$',
    fs=10.5,w='bold')

# ═════════ PANEL (b) ═════════
panel(5.10,4.36,'b'); txt(5.43,4.36,'Gram matrix and classifier',fs=11,w='bold',ha='left')

n=26; half=n//2
Xc=np.vstack([np.random.randn(half,2)*0.5+[1.5,0],np.random.randn(half,2)*0.5+[-1.5,0]])
D2=((Xc[:,None,:]-Xc[None,:,:])**2).sum(-1)
Kb=np.exp(-0.5*D2); np.fill_diagonal(Kb,1.0)
ax.imshow(Kb,extent=(4.65,6.15,1.65,3.15),cmap='inferno',vmin=0,vmax=1,zorder=3)
ax.add_patch(Rectangle((4.65,1.65),1.5,1.5,fill=False,edgecolor=BRN,lw=1.5,zorder=4))
txt(5.40,3.38,r'$K_Q$ over training pairs (schematic)',fs=9.5,c='#000000')
txt(5.40,1.42,r'$K_Q\in[0,1]^{N\times N},\ \ \mathrm{diag}=1$',fs=8.8)

arr(6.25,2.40,6.75,2.40,c=DARK,lw=1.8)

bx0,by0,bw,bh=6.85,1.75,1.55,1.45
ax.add_patch(Rectangle((bx0,by0),bw,bh,facecolor='#fbfbfd',edgecolor=DARK,lw=1.2,zorder=2))
qa=np.random.randn(8,2)*[0.16,0.16]+[bx0+0.42,by0+1.02]
qb=np.random.randn(8,2)*[0.16,0.16]+[bx0+1.10,by0+0.44]
ax.scatter(qa[:,0],qa[:,1],s=17,color=C_A,zorder=4,edgecolor='white',lw=0.4)
ax.scatter(qb[:,0],qb[:,1],s=17,color=C_B,zorder=4,edgecolor='white',lw=0.4)
tt=np.linspace(0,1,60)
bxs=bx0+0.12+1.32*tt; bys=by0+1.28-1.05*tt+0.28*np.sin(3.1*tt)
ax.plot(bxs,bys,color=GRN,lw=1.8,zorder=5)
txt(bx0+bw/2,by0+bh+0.18,'SVM on precomputed $K_Q$',fs=8.8)
txt(6.60,0.62,r'$f(x)=\mathrm{sgn}\!\left(\sum_{i}\alpha_i\, y_i\, K_Q(x_i,x)+b\right)$',fs=9.5,c="#000000")

# ═════════ PANEL (c) ═════════
panel(10.05,4.36,'c'); txt(10.38,4.36,'Concentration',fs=11,w='bold',ha='left')

s1x,s1y,s1r=9.72,3.47,0.55
sphere(s1x,s1y,s1r,lw=1.2)
for a0 in [52,64,76]: statevec(s1x,s1y,s1r,a0,C_A,lw=1.5)
for a0 in [128,140,152]: statevec(s1x,s1y,s1r,a0,C_B,lw=1.5)
txt(s1x,s1y-s1r-0.18,'states cluster by class',fs=8.2,c='#000000')
arr(10.42,3.47,10.78,3.47,c=DARK,lw=1.5)
ax.imshow(Kb,extent=(10.86,11.76,3.02,3.92),cmap='inferno',vmin=0,vmax=1,zorder=3)
ax.add_patch(Rectangle((10.86,3.02),0.9,0.9,fill=False,edgecolor=GRN,lw=1.4,zorder=4))
txt(12.14,3.47,'class blocks\nretained',fs=8.6,c=GRN)

s2x,s2y,s2r=9.72,1.67,0.55
sphere(s2x,s2y,s2r,lw=1.2)
for a0 in [15,58,103,148,196,241,286,331]: statevec(s2x,s2y,s2r,a0,'#8a8a8a',lw=1.3)
txt(s2x+0.85,s2y-s2r-0.18,'states nearly orthogonal',fs=8.2,c='#000000')
arr(10.42,1.67,10.78,1.67,c=DARK,lw=1.5)
Kid=np.eye(n)+np.abs(np.random.randn(n,n))*0.03
Kid=np.clip((Kid+Kid.T)/2,0,1); np.fill_diagonal(Kid,1.0)
ax.imshow(Kid,extent=(10.86,11.76,1.22,2.12),cmap='inferno',vmin=0,vmax=1,zorder=3)
ax.add_patch(Rectangle((10.86,1.22),0.9,0.9,fill=False,edgecolor=BRN,lw=1.4,zorder=4))
txt(12.13,1.67,r'$K_Q\approx I$',fs=9,c=BRN)
txt(10.55,0.67,r'$\langle\phi(x)|\phi(x^{\prime})\rangle\to 0$ as states spread over $\mathbb{C}^{2^{n}}$',fs=9.5,c='#000000')

cax=fig.add_axes([4.75/W,(1.00-Y0)/(H-Y0),1.20/W,0.09/(H-Y0)])
import matplotlib as _mpl
_cb=_mpl.colorbar.ColorbarBase(cax,cmap='inferno',norm=_mpl.colors.Normalize(0,1),orientation='horizontal',ticks=[0,0.5,1])
cax.tick_params(labelsize=7*FS,length=2,pad=1.5); cax.set_title(r'$K(x,x^{\prime})$',fontsize=7.5*FS,pad=2)
for ext in ('pdf','png','svg'):
    fig.savefig(f'{OUT}/fig2_qkernel.{ext}',dpi=200,bbox_inches='tight',pad_inches=0.05)
print('saved')
