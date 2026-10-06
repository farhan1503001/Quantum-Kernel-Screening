import os as _os
DATA = _os.environ.get('QKS_DATA_DIR', _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data'))
OUT  = _os.environ.get('QKS_FIG_DIR', _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'  # keep text as text in SVG
plt.rcParams['font.weight']='medium'
plt.rcParams['axes.labelweight']='medium'
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Polygon, Rectangle

np.random.seed(3)
W, H = 12.5, 7.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')

# palette (Okabe-Ito based)
VIO   = '#b45309'; VIO_BG = '#fbeed3'    # quantum (paper brown)
TEA   = '#1f7a3d'; TEA_BG = '#e3f2e5'    # classical (paper green)
DARK  = '#111827'; GRAY = '#9ca3af'
C_TX  = '#0072B2'; C_VI = '#E69F00'; C_MM = '#009E73'; C_ML = '#CC79A7'
GREEN = '#15803d'; RED = '#b2182b'

def rbox(x,y,w,h,fc,ec,lw=1.4,r=0.10):
    b=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.045,rounding_size={r}",
                     facecolor=fc,edgecolor=ec,linewidth=lw,zorder=2); ax.add_patch(b); return b
def txt(x,y,s,fs=10.4,c=DARK,w='medium',ha='center',va='center',st='normal',z=4):
    ax.text(x,y,s,fontsize=fs,color=c,fontweight=w,ha=ha,va=va,style=st,zorder=z)
def arr(x1,y1,x2,y2,c=DARK,lw=1.6,style='-|>',ls='-',cs=None,z=3):
    a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=13,color=c,
                      linewidth=lw,linestyle=ls,zorder=z,connectionstyle=cs); ax.add_patch(a)
def panel(x,y,s):
    ax.add_patch(Circle((x+0.14, y), 0.15, facecolor=DARK, edgecolor='none', zorder=5))
    ax.text(x+0.14, y, s, fontsize=11.5, color='white', fontweight='bold',
            ha='center', va='center', zorder=6)

# ════════ PANEL (a): PIPELINE  (y band 4.75 .. 7.15) ════════
panel(0.22, 7.08, 'a')
txt(0.55, 7.08, 'Frozen-embedding pipeline: one input, two kernels', fs=11.4, w='bold', ha='left')

# --- input data cards ---
cardx, cw, chh = 0.38, 0.86, 0.52
for i,(lab,yy) in enumerate([('Aa',6.30),('img',5.62),('xx',4.94)]):
    rbox(cardx,yy,cw,chh,'#ffffff',GRAY,lw=1.1,r=0.07)
    if lab=='Aa': txt(cardx+cw/2,yy+chh/2,'Aa',fs=12.0,w='bold',c=DARK)
    elif lab=='img':
        for r_ in range(3):
            for c_ in range(3):
                ax.add_patch(Rectangle((cardx+0.18+c_*0.17, yy+0.08+r_*0.13),0.14,0.10,
                    facecolor=plt.cm.viridis(0.15+0.1*(r_*3+c_)/9+0.35*np.random.rand()),zorder=3))
    else:
        c0=Circle((cardx+cw/2,yy+chh/2),0.19,facecolor='none',edgecolor=DARK,lw=1.2,zorder=3); ax.add_patch(c0)
        ax.add_patch(Arc((cardx+cw/2,yy+chh/2),0.20,0.38,edgecolor=DARK,lw=0.9,zorder=3))
        ax.plot([cardx+cw/2-0.19,cardx+cw/2+0.19],[yy+chh/2]*2,color=DARK,lw=0.9,zorder=3)
txt(cardx+cw/2, 4.72, 'text · vision\nmultimodal · 6 languages', fs=7.9, c='#3a3a3a')

# --- frozen encoder stack ---
ex, ey, ew = 1.72, 5.02, 1.42
for i in range(4):
    rbox(ex, ey+i*0.42, ew, 0.34, '#eef2f7', '#64748b', lw=1.2, r=0.06)
# neurons inside each layer + faint inter-layer connections (network motif)
dot_xs = [ex + ew*f for f in (0.18, 0.40, 0.62, 0.84)]
dot_ys = [ey + i*0.42 + 0.17 for i in range(4)]
for i in range(3):
    for xa in dot_xs:
        for xb in dot_xs:
            ax.plot([xa, xb], [dot_ys[i]+0.03, dot_ys[i+1]-0.03],
                    color='#94a3b8', lw=0.35, alpha=0.45, zorder=3)
for yy in dot_ys:
    for xx in dot_xs:
        ax.add_patch(Circle((xx, yy), 0.036, facecolor='#475569', zorder=4))
txt(ex+ew/2, ey-0.20, 'Frozen encoder', fs=10.2, w='bold')
txt(ex+ew/2+0.15, ey-0.50, 'BERT · MiniLM · RoBERTa · CLIP · DINOv2\nViT · SigLIP · FLAVA · mBERT · XLM-R', fs=6.4, c='#3a3a3a')
# snowflake (drawn)
# frozen badge pinned to the stack's top-right corner
sx, sy = ex+ew-0.06, ey+3*0.42+0.34
ax.add_patch(Circle((sx, sy), 0.155, facecolor='#ffffff', edgecolor='#3b82f6', lw=1.4, zorder=5))
sr = 0.095
for ang in range(0,180,60):
    a_=np.deg2rad(ang)
    ax.plot([sx-sr*np.cos(a_),sx+sr*np.cos(a_)],[sy-sr*np.sin(a_),sy+sr*np.sin(a_)],
            color='#3b82f6',lw=1.5,zorder=6)
arr(cardx+cw+0.05, 5.85, ex-0.06, 5.85)

# --- embedding column ---
emx, emy = 3.62, 5.10
cmap_cells = plt.cm.plasma(np.linspace(0.15,0.85,9))
for i in range(9):
    ax.add_patch(Rectangle((emx, emy+i*0.17), 0.26, 0.15, facecolor=cmap_cells[i], zorder=3))
txt(emx+0.13, emy-0.22, 'embedding', fs=8.5, c='#3a3a3a')
arr(ex+ew+0.05, 5.85, emx-0.05, 5.85)

# --- PCA box ---
rbox(4.28, 5.62, 0.98, 0.5, '#ffffff', DARK, lw=1.2)
txt(4.77, 5.87, r'PCA$\rightarrow$8', fs=10.2)
arr(emx+0.31, 5.87, 4.23, 5.87)

# --- branch: quantum (top) & classical (bottom) ---
arr(5.31, 5.95, 5.72, 6.34, c=VIO)
arr(5.31, 5.79, 5.72, 5.28, c=TEA)
rbox(5.78, 6.10, 2.0, 0.62, VIO_BG, VIO, lw=1.5)
txt(6.78, 6.47, 'Quantum feature map', fs=10.0, c=VIO, w='bold')
txt(6.78, 6.26, r'$|\phi(x)\rangle=U(x)\,|0\rangle^{\otimes 8}$', fs=9.3, c=VIO)
rbox(5.78, 4.98, 2.0, 0.62, TEA_BG, TEA, lw=1.5)
txt(6.78, 5.35, 'RBF kernel', fs=10.0, c=TEA, w='bold')
txt(6.78, 5.14, r'$k(x,x^{\prime})=e^{-\gamma\|x-x^{\prime}\|^{2}}$', fs=9.3, c=TEA)

# --- mini kernel heatmaps in pipeline ---
n=26; half=n//2
Xc = np.vstack([np.random.randn(half,2)*0.55+[1.6,0], np.random.randn(half,2)*0.55+[-1.6,0]])
D2 = ((Xc[:,None,:]-Xc[None,:,:])**2).sum(-1)
K_rbf = np.exp(-0.55*D2)
K_q = np.eye(n) + np.abs(np.random.randn(n,n))*0.030; K_q=np.clip((K_q+K_q.T)/2,0,1); np.fill_diagonal(K_q,1)
ax.imshow(K_q, extent=(8.05,8.85,6.06,6.86), cmap='inferno', vmin=0, vmax=1, zorder=3)
ax.add_patch(Rectangle((8.05,6.06),0.8,0.8,fill=False,edgecolor=VIO,lw=1.4,zorder=4))
ax.imshow(K_rbf, extent=(8.05,4.94-0.0+4.94-0.94,None,None) if False else (8.05,8.85,4.94,5.74), cmap='inferno', vmin=0, vmax=1, zorder=3)
ax.add_patch(Rectangle((8.05,4.94),0.8,0.8,fill=False,edgecolor=TEA,lw=1.4,zorder=4))
txt(8.45, 5.95, r'$K_Q$', fs=9.7, c=VIO); txt(8.45, 4.80, r'$K_C$', fs=9.7, c=TEA)
arr(7.83, 6.41, 8.00, 6.44, c=VIO); arr(7.83, 5.29, 8.00, 5.32, c=TEA)

# --- SVMs + gap ---
rbox(9.22, 6.16, 1.62, 0.56, VIO_BG, VIO, lw=1.4); txt(10.03, 6.47, 'Quantum SVM', fs=10.0, c=VIO, w='bold')
rbox(9.22, 5.02, 1.62, 0.56, TEA_BG, TEA, lw=1.4); txt(10.03, 5.33, 'Classical SVM', fs=10.0, c=TEA, w='bold')
arr(8.90, 6.46, 9.17, 6.46, c=VIO); arr(8.90, 5.34, 9.17, 5.31, c=TEA)
# gap bracket
ax.plot([11.02,11.14,11.14,11.02],[6.44,6.44,5.30,5.30],color=DARK,lw=1.3,zorder=3)
txt(11.30, 5.88, r'$\Delta = \mathrm{F1}_Q-\mathrm{F1}_C$', fs=10.2, ha='left', w='bold')
txt(11.30, 5.58, 'predicted in advance,\nsee (d)', fs=8.7, ha='left', c=GREEN, st='italic')

# ════════ PANEL (b): IQP CIRCUIT (x 0.3..4.45, y 0.4..4.1) ════════
panel(0.22, 4.28, 'b')
txt(0.55, 4.28, 'IQP feature map (8 qubits, 4 shown)', fs=10.9, w='bold', ha='left')
wy = [3.55, 2.95, 2.35, 1.75]
cx0, cx1 = 0.70, 4.03
for y_ in wy:
    ax.plot([cx0, cx1],[y_,y_],color=DARK,lw=1.1,zorder=1)
    txt(cx0-0.26, y_, r'$|0\rangle$', fs=9.7)
def gate(x,y,label,fc='#ffffff',ec=DARK,w_=0.34,h_=0.30,fs=8.7):
    rbox(x-w_/2, y-h_/2, w_, h_, fc, ec, lw=1.1, r=0.04); txt(x,y,label,fs=fs)
# H layer
for y_ in wy: gate(1.03, y_, 'H', fc='#d9e9f8', ec='#4a7fb5')
# RZ(x_i) layer
for i,y_ in enumerate(wy): gate(1.61, y_, r'$R_Z$', fc=VIO_BG, ec=VIO, fs=8.7)
txt(1.61, wy[0]+0.32, r'$R_Z(x_i)$', fs=8.2, c=VIO)
# entanglers ZZ(x_i x_j) between adjacent wires
for i in range(3):
    xz = 2.23 + i*0.42
    ax.plot([xz,xz],[wy[i],wy[i+1]],color=VIO,lw=1.3,zorder=2)
    ax.add_patch(Circle((xz,wy[i]),0.045,facecolor=VIO,zorder=3))
    ax.add_patch(Circle((xz,wy[i+1]),0.045,facecolor=VIO,zorder=3))
    gate(xz,(wy[i]+wy[i+1])/2, 'ZZ', fc=VIO_BG, ec=VIO, w_=0.30,h_=0.24, fs=7.7)
txt(2.65, wy[0]+0.32, r'$ZZ(x_i x_{i+1})$', fs=8.2, c=VIO)
# adjoint block
rbox(3.37, wy[3]-0.20, 0.44, wy[0]-wy[3]+0.40, '#ffffff', VIO, lw=1.5)
txt(3.59, (wy[0]+wy[3])/2, r'$U^{\dagger}(x^{\prime})$', fs=9.7, c=VIO)
# measurements
for y_ in wy:
    mx = 4.19
    rbox(mx-0.14, y_-0.14, 0.30, 0.28, '#e2f3e2', '#217a3c', lw=1.1, r=0.03)
    ax.add_patch(Arc((mx+0.01,y_-0.06),0.20,0.20,theta1=0,theta2=180,edgecolor='#217a3c',lw=1.1,zorder=4))
    ax.plot([mx+0.01,mx+0.09],[y_-0.06,y_+0.08],color='#217a3c',lw=1.1,zorder=4)
txt(2.35, 1.18, r'$K_Q(x,x^{\prime}) = \left|\langle 0|U^{\dagger}(x^{\prime})\,U(x)|0\rangle\right|^{2}$',
    fs=10.4, c=DARK)
txt(2.35, 0.72, '83 min per 1000-point kernel matrix (simulator)', fs=8.9, c=RED, st='italic')

# ════════ PANEL (c): CONCENTRATION (x 4.8..8.35) ════════
panel(4.95, 4.28, 'c')
txt(5.28, 4.28, 'Concentration destroys structure', fs=10.9, w='bold', ha='left')
ax.imshow(K_rbf, extent=(5.05,6.45,2.15,3.55), cmap='inferno', vmin=0, vmax=1, zorder=3)
ax.add_patch(Rectangle((5.05,2.15),1.4,1.4,fill=False,edgecolor=TEA,lw=1.6,zorder=4))
ax.imshow(K_q, extent=(6.85,8.25,2.15,3.55), cmap='inferno', vmin=0, vmax=1, zorder=3)
ax.add_patch(Rectangle((6.85,2.15),1.4,1.4,fill=False,edgecolor=VIO,lw=1.6,zorder=4))
txt(5.75, 3.72, 'RBF kernel', fs=9.7, c=TEA, w='bold')
txt(7.55, 3.72, 'Quantum kernel', fs=9.7, c=VIO, w='bold')
txt(5.75, 1.92, r'mean off-diag $\approx 0.55$', fs=8.9, c=TEA)
txt(7.55, 1.92, r'mean off-diag $\approx 0.04$', fs=8.9, c=VIO)
txt(5.75, 1.66, 'class blocks visible', fs=8.7, c='#3a3a3a', st='italic')
txt(7.55, 1.66, 'near identity', fs=8.7, c='#3a3a3a', st='italic')
txt(6.65, 1.12, 'alignment (KTA) measures whether the\nsurviving geometry still matches the labels', fs=9.3, c=DARK)
# colorbar
cax = fig.add_axes([5.05/W, 0.52/H, 3.2/W, 0.10/H])
import matplotlib as mpl
mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.inferno, orientation='horizontal',
                          norm=mpl.colors.Normalize(0,1))
cax.tick_params(labelsize=6.5); cax.set_xlabel(r'$K(x,x^{\prime})$', fontsize=8.2)

# ════════ PANEL (d): THE SCREEN — 18 main-grid configurations, classical alignment ════════
import json
from scipy import stats as _st
panel(8.68, 4.28, 'd')
txt(9.0, 4.28, 'Zero-shot screen (18 configurations)', fs=10.9, w='bold', ha='left')
sax = fig.add_axes([9.02/W, 0.95/H, 3.18/W, 2.95/H])
_main = json.load(open(_os.path.join(DATA, 'grid_split.json')))['main']
kta = np.array([c['c_kta'] for c in _main]); gap = np.array([c['gap'] for c in _main]); dom = [c['domain'] for c in _main]
cmap_d={'text':C_TX,'vision':C_VI,'multimodal':C_MM}; mk_d={'text':'o','vision':'s','multimodal':'^'}
sax.axvspan(0, 0.05, color='#fdecea', zorder=0)
sax.text(0.047, -0.0745, 'skip\nquantum', fontsize=8.2, color=RED, ha='right', style='italic')
for d in ['text','vision','multimodal']:
    m=[i for i,dd in enumerate(dom) if dd==d]
    sax.scatter(kta[m], gap[m], s=36, marker=mk_d[d], color=cmap_d[d], edgecolor='white', lw=0.7, label=d, zorder=3)
z=np.polyfit(kta,gap,1); xs=np.linspace(0.0,0.86,60)
sax.plot(xs, np.poly1d(z)(xs), '--', color='#555555', lw=1.4, zorder=2)
sax.axvline(0.05, color=RED, lw=1.0, ls=':')
sax.axhline(0, color='black', lw=0.7, alpha=0.5)
sax.set_xlabel('kernel-target alignment (classical, free)', fontsize=9.2)
sax.set_ylabel(r'gap $\Delta=\mathrm{F1}_Q-\mathrm{F1}_C$', fontsize=9.2)
sax.set_xlim(-0.02, 0.88); sax.set_ylim(-0.078, 0.02)
sax.tick_params(labelsize=7.5)
sax.legend(fontsize=7.5, loc='lower right', framealpha=0.9, handletextpad=0.15)
_r=_st.pearsonr(kta,gap)
sax.text(0.30, -0.060, r'$r=0.87$'+'\n'+r'$p=2\times10^{-6}$', fontsize=9.6, color=DARK, fontweight='bold')
for s in ['top','right']: sax.spines[s].set_visible(False)
print(f"panel d: n={len(kta)} r={_r[0]:.3f} p={_r[1]:.1e}")

for ext in ('pdf','png','svg'):
    fig.savefig(f'{OUT}/fig1_concept.{ext}', dpi=200, bbox_inches='tight')
print('saved')
