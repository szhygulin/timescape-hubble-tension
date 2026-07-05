#!/usr/bin/env python3
"""Decomposition figure: dBIC across magnitude x covariance treatment.
Shows that the timescape preference appears ONLY in the (raw-Tripp, diagonal)
corner; the full covariance favours LCDM for every magnitude definition."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fit_timescape as F
OUT=".."; A0,B0=0.131,2.663

with open(F.DATA) as f:
    h=f.readline().split(); idx={n:i for i,n in enumerate(h)}; rows=[l.split() for l in f]
col=lambda n:np.array([float(r[idx[n]]) for r in rows])
d={n:col(n) for n in ["zHD","zHEL","m_b_corr","mB","x1","c","biasCor_m_b"]}
iscal=np.array([int(float(r[idx["IS_CALIBRATOR"]])) for r in rows]); m=(iscal==0)&(d["zHD"]>0.01)
d={k:v[m] for k,v in d.items()}
with open(F.COV) as f: n=int(f.readline())
Cfull=np.loadtxt(F.COV,skiprows=1).reshape(n,n)[np.ix_(m,m)]
Cdiag=np.diag(np.diag(Cfull))
zHD,zHEL=d["zHD"],d["zHEL"]
mags={"m_b_corr (standard)":d["m_b_corr"],"raw Tripp (no bias corr.)":d["mB"]+A0*d["x1"]-B0*d["c"]}

def dbic(mag,C):
    Cinv=np.linalg.inv(C); one=np.ones(len(mag)); s11=one@(Cinv@one)
    def chi2(Ds):
        r=mag-5*np.log10((1+zHEL)*Ds); Cr=Cinv@r; return r@Cr-(one@Cr)**2/s11
    _,cT,*_=F.scan(chi2,lambda fv:F.D_shape_TS(zHD,fv),np.linspace(0.5,0.9,400),"")
    _,cL,*_=F.scan(chi2,lambda om:F.D_shape_LCDM(zHD,om),np.linspace(0.05,0.6,300),"")
    return cT-cL

res={}
for mname,mag in mags.items():
    for cname,C in [("Full covariance",Cfull),("Diagonal\n(off-diag dropped)",Cdiag)]:
        res[(mname,cname)]=float(dbic(mag,C))
        print(mname,"|",cname.replace("\n"," "),"-> dBIC",round(res[(mname,cname)],2))
# bias-correction-removed full-covariance cell (m_b_corr + biasCor_m_b undoes the
# BBC bias correction); reported in Table I / the abstract as +9.0. Written to the
# JSON only (not shown as a figure bar, which keeps the two main magnitudes).
_bias_rm=float(dbic(d["m_b_corr"]+d["biasCor_m_b"],Cfull))
_out={f"{a} | {b}":v for (a,b),v in res.items()}
_out["bias correction removed | Full covariance"]=_bias_rm
_out["bias_correction_removed_note"]=("fv0 rails to the fvg grid edge (0.900); "
    "source: m_b_corr + biasCor_m_b under the full stat+sys covariance -- "
    "reproduces the abstract's/Table I's +9.0")
json.dump(_out,open(f"{OUT}/results_decomp.json","w"),indent=2)
print("bias correction removed | Full covariance -> dBIC",round(_bias_rm,2))

# grouped bar
covs=["Full covariance","Diagonal\n(off-diag dropped)"]; mnames=list(mags)
x=np.arange(len(covs)); w=0.36
fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.axhline(0,color="k",lw=0.8)
for j,mn in enumerate(mnames):
    vals=[res[(mn,c)] for c in covs]
    cols=["#c0392b" if v>0 else "#2471a3" for v in vals]
    bars=ax.bar(x+(j-0.5)*w,vals,w,color=cols,edgecolor="k",lw=0.6,
                hatch=("" if j==0 else "//"),label=mn)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,v+(0.3 if v>0 else -0.6),f"{v:+.1f}",
                ha="center",va="bottom" if v>0 else "top",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(covs); ax.set_ylabel(r"$\Delta$BIC (timescape $-\ \Lambda$CDM)")
ax.text(0.015,0.96,"← favours timescape",transform=ax.transAxes,color="#2471a3",va="top",fontsize=9)
ax.text(0.015,0.05,"favours $\\Lambda$CDM ↑",transform=ax.transAxes,color="#c0392b",va="bottom",fontsize=9)
ax.set_title("Timescape is preferred only when the off-diagonal covariance is dropped",fontsize=10)
ax.legend(title="apparent magnitude",fontsize=8.5,loc="upper right")
ax.set_ylim(min(res.values())-3,max(res.values())+3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_dbic.pdf"); fig.savefig(f"{OUT}/fig_dbic.png",dpi=150)
print("wrote fig_dbic.pdf (decomposition)")
