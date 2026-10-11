"""
Reproduce Figs. 2, 3 and 4 of the paper from results/floors.json.

Usage:  python src/make_figures.py [--floors results/floors.json] [--out figures]
"""
import argparse, os
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, LogLocator
ap=argparse.ArgumentParser(); ap.add_argument('--floors',default='results/floors.json'); ap.add_argument('--out',default='figures')
A_=ap.parse_args(); os.makedirs(A_.out,exist_ok=True)
D=json.load(open(A_.floors))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'legend.fontsize':7.8,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.8,'lines.linewidth':1.4,'lines.markersize':4.5,
 'savefig.bbox':'tight','savefig.pad_inches':0.03})
C_HB,C_POD,C_DMD='#D55E00','#0072B2','#009E73'
plain=FuncFormatter(lambda v,_: (f'{v:g}'))
def logax(ax,lo,hi):
    ax.set_yscale('log'); ax.set_ylim(lo,hi)
    dec=[10.0**e for e in range(int(np.floor(np.log10(lo))),int(np.ceil(np.log10(hi)))+1)]
    ax.yaxis.set_major_locator(FixedLocator([d for d in dec if lo<=d<=hi]))
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2,10))); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True,which='major',alpha=0.3,lw=0.5); ax.grid(True,which='minor',alpha=0.12,lw=0.4)

# ---------------- Fig 2: three cases, shared axis
cases=[('fsi2','(a) FSI2'),('fsi3','(b) FSI3'),('cylinder','(c) Cylinder')]
fig,axs=plt.subplots(1,3,figsize=(7.0,2.55),sharey=True)
for ax,(c,lab) in zip(axs,cases):
    d=D[c]['fig2']; K=d['K']
    ax.plot(K,d['POD_opt'],'o-',color=C_POD,label=r'POD, optimal floor $\varepsilon^*_K$')
    ax.plot(K,d['HB_ref'],'s-',color=C_HB,label=r'HB at refined $f_0^*$')
    ax.plot(K,d['HB_peak'],'s--',color=C_HB,alpha=0.55,mfc='none',label=r'HB at spectral peak $\hat f$')
    ax.plot(K,d['DMD'],'^',color=C_DMD,ms=7.5,mfc='none',mew=1.4,ls='none',zorder=6,label='DMD')
    logax(ax,0.1,30); ax.set_xticks(K); ax.set_xlabel('temporal functions $K$')
    ax.set_title(f"{lab}, $N$ = {D[c]['N']:.0f}",loc='left',fontsize=9)
axs[0].set_ylabel('pressure floor, all fluid (%)')
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(0.5,-0.02),columnspacing=1.4,handlelength=2.2)
fig.tight_layout(rect=(0,0.06,1,1))
for e in ('png','pdf'): fig.savefig(os.path.join(A_.out,f'fig2_floors.{e}'),dpi=300)
plt.close(fig)

# ---------------- Fig 3: sensitivity, case colours, Eq. prediction as curves
def pred(w,N,d,NH=3):
    s=lambda x: 1.0 if x==0 else np.sin(x)/x
    e2=sum(w[str(k)]*(1-s(np.pi*k*N*d)**2) for k in range(1,NH+1))
    e2+=sum(w[str(k)] for k in range(NH+1,9))+w['rest']
    return np.sqrt(max(e2,0))*100
CC={'fsi2':('#000000','o','FSI2'),'fsi3':('#6E6E6E','s','FSI3'),'cylinder':('#7B3294','^','Cylinder')}
fig,ax=plt.subplots(figsize=(4.6,3.0))
dd=np.linspace(0,3.0,301)
for c,(col,mk,lab) in CC.items():
    r=D[c]; w=r['energy']; N=r['N']
    ax.plot(dd,[pred(w,N,x/100) for x in dd],'-',color=col,lw=1.1)
    ax.plot(r['fig3a']['err'],r['fig3a']['floor'],mk,color=col,mfc='white',mew=1.1,ls='none',
            label=f'{lab}, $N$ = {N:.0f}')
    ax.plot([r['eps_peak']],[r['fig3a']['peak_floor']],'*',ms=12,color=col,mec='k',mew=0.4)
    ax.axvline(r['worst'],color=col,ls=':',lw=1.0)
logax(ax,0.2,100); ax.set_xlim(-0.05,3.05)
ax.set_xlabel(r'relative error in assumed fundamental, $\delta$ (%)')
ax.set_ylabel(r'surface-pressure floor at $N_H$ = 3 (%)')
from matplotlib.lines import Line2D
extra=[Line2D([],[],color='0.3',lw=1.1,label='Eq. (9)'),
       Line2D([],[],marker='*',ms=11,color='0.6',mec='k',mew=0.4,ls='none',label='spectral peak-picking'),
       Line2D([],[],color='0.3',ls=':',lw=1.0,label=r'worst case $1/(2N)$')]
h,l=ax.get_legend_handles_labels()
ax.legend(h+extra,l+[e.get_label() for e in extra],frameon=False,loc='lower right',fontsize=7.4,ncol=2,columnspacing=1.0)
fig.tight_layout()
for e in ('png','pdf'): fig.savefig(os.path.join(A_.out,f'fig3_sensitivity.{e}'),dpi=300)
plt.close(fig)

# ---------------- Fig 4: optimal floors by region
sty={'interface':('-','o','#08306B'),'near-body':('--','s','#2171B5'),'wake':(':','^','#6BAED6'),'all fluid':('-.','d','#9E9E9E')}
fig,axs=plt.subplots(1,2,figsize=(6.9,2.6),sharey=True)
for ax,(c,lab) in zip(axs,[('fsi2','(a) FSI2'),('fsi3','(b) FSI3')]):
    for r,ys in D[c]['fig4'].items():
        ls,mk,col=sty[r]; ax.plot(D[c]['fig2']['K'],ys,ls=ls,marker=mk,color=col,label=f"{r} ({D[c]['nreg'][r]} nodes)")
    logax(ax,0.005,60); ax.set_xticks(D[c]['fig2']['K']); ax.set_xlabel('temporal functions $K$')
    ax.set_title(lab,loc='left',fontsize=9)
axs[0].set_ylabel(r'optimal pressure floor $\varepsilon^*_K$ (%)')
axs[1].legend(frameon=False,loc='lower left',fontsize=7.4)
fig.tight_layout()
for e in ('png','pdf'): fig.savefig(os.path.join(A_.out,f'fig4_regions.{e}'),dpi=300)
plt.close(fig)
from PIL import Image
for f in ('fig2_floors.png','fig3_sensitivity.png','fig4_regions.png'): print(os.path.join(A_.out,f),Image.open(os.path.join(A_.out,f)).size)
