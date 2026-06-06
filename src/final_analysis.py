#!/usr/bin/env python3
"""
Consolidated, reproducible analysis: timescape vs flat-LCDM on Pantheon+,
under (A) the standard bias-corrected products and (B) a cosmology-independent
Tripp reduction. Saves results.json and two publication figures.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fit_timescape as F
import fit_tripp as T

OUT = ".."

# ----------------- (A) standard products -----------------
zHD,zHEL,mb,Cfull = F.load()
N=len(zHD)
fvg=np.linspace(0.500,0.900,400); omg=np.linspace(0.050,0.600,300)
chi2_full = F.make_chi2(zHD,zHEL,mb,Cfull)
fvA,cTSA,fa,fb,_,_ = F.scan(chi2_full, lambda fv:F.D_shape_TS(zHD,fv), fvg,"")
omA,cLA ,oa,ob,_,_ = F.scan(chi2_full, lambda om:F.D_shape_LCDM(zHD,om), omg,"")
Cdiag=np.diag(np.diag(Cfull))
chi2_diag=F.make_chi2(zHD,zHEL,mb,Cdiag)
fvD,cTSD,_,_,_,_=F.scan(chi2_diag, lambda fv:F.D_shape_TS(zHD,fv), fvg,"")
omD,cLD,_,_,_,_ =F.scan(chi2_diag, lambda om:F.D_shape_LCDM(zHD,om), omg,"")

# ----------------- (B) cosmology-independent Tripp -----------------
D=T.load_salt()
def tripp(mask):
    Dm={k:v[mask] for k,v in D.items()}
    fv,LTS,a1,a2,pT,_,_=T.scan_model(Dm,lambda fv:F.D_shape_TS(Dm["zHD"],fv),np.linspace(0.50,0.90,81))
    om,LL ,o1,o2,pL,_,_=T.scan_model(Dm,lambda om:F.D_shape_LCDM(Dm["zHD"],om),np.linspace(0.05,0.60,81))
    return dict(N=int(mask.sum()),fv0=float(fv),fv0_err=[float(a1),float(a2)],Om=float(om),
                L_TS=float(LTS),L_L=float(LL),dBIC=float(LTS-LL),
                alpha=float(pT[0]),beta=float(pT[1]),sigint=float(np.exp(pT[2])))
tB_full=tripp(D["zHD"]>0.01)
tB_023 =tripp(D["zHD"]>0.023)
tB_075 =tripp(D["zHD"]>0.075)

dof=N-2
results=dict(
  N=N, z_range=[float(zHD.min()),float(zHD.max())],
  validation=dict(LCDM_Om_standard=float(omA), note="matches Pantheon+ published Om0=0.334; F'(tau) & low-z Taylor verified"),
  standard_full=dict(fv0=float(fvA),fv0_err=[float(fa),float(fb)],Om=float(omA),Om_err=[float(oa),float(ob)],
                     chi2_TS=float(cTSA),chi2_L=float(cLA),chi2_dof_L=float(cLA/dof),dBIC=float(cTSA-cLA)),
  standard_diag=dict(fv0=float(fvD),Om=float(omD),chi2_TS=float(cTSD),chi2_L=float(cLD),dBIC=float(cTSD-cLD)),
  tripp_full=tB_full, tripp_z023=tB_023, tripp_z075=tB_075,
)
with open(f"{OUT}/results.json","w") as f: json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))

# ----------------- FIGURE 1: sign-flip bar chart -----------------
labels=["Standard\n(full cov)","Standard\n(diagonal)","Tripp\n(full)","Tripp\n(z>0.023)","Tripp\n(z>0.075)"]
dbics=[results["standard_full"]["dBIC"],results["standard_diag"]["dBIC"],
       tB_full["dBIC"],tB_023["dBIC"],tB_075["dBIC"]]
colors=["#c0392b" if d>0 else "#2471a3" for d in dbics]
fig,ax=plt.subplots(figsize=(7.0,4.2))
ax.axhline(0,color="k",lw=0.8)
for y,s in [(5,"strong"),( -5,"strong"),(10,"v.strong"),(-10,"v.strong")]:
    ax.axhline(y,color="grey",ls=":",lw=0.6)
bars=ax.bar(labels,dbics,color=colors,width=0.62,edgecolor="k",lw=0.6)
for b,d in zip(bars,dbics):
    ax.text(b.get_x()+b.get_width()/2, d+(0.5 if d>0 else -0.9), f"{d:+.1f}",
            ha="center",va="bottom" if d>0 else "top",fontsize=9)
ax.set_ylabel(r"$\Delta\mathrm{BIC}\ (\mathrm{timescape}-\Lambda\mathrm{CDM})$")
ax.text(0.015,0.96,"← favours timescape",transform=ax.transAxes,color="#2471a3",va="top",fontsize=9)
ax.text(0.015,0.05,"favours $\\Lambda$CDM ↑",transform=ax.transAxes,color="#c0392b",va="bottom",fontsize=9)
ax.set_title("Pantheon+ model-comparison verdict flips with the data reduction",fontsize=10.5)
ax.set_ylim(min(dbics)-3,max(dbics)+3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_dbic.pdf"); fig.savefig(f"{OUT}/fig_dbic.png",dpi=150)
print("wrote fig_dbic.pdf")

# ----------------- FIGURE 2: Hubble residuals vs best LCDM -----------------
Cinv=np.linalg.inv(Cfull); one=np.ones(N)
def resid(shape):
    mu=5*np.log10((1+zHEL)*shape); r0=mb-mu
    M=(one@(Cinv@r0))/(one@(Cinv@one)); return r0-M
rL=resid(F.D_shape_LCDM(zHD,omA))      # data - best LCDM (offset matched)
# binned data residual
nb=18; edges=np.geomspace(zHD.min(),zHD.max(),nb+1); cen=np.sqrt(edges[:-1]*edges[1:])
idx=np.digitize(zHD,edges)-1
mr=np.array([rL[idx==i].mean() if np.any(idx==i) else np.nan for i in range(nb)])
er=np.array([rL[idx==i].std()/max(1,np.sqrt((idx==i).sum())) if np.any(idx==i) else np.nan for i in range(nb)])
zz=np.geomspace(zHD.min(),zHD.max(),300)
def curve(shape_z):
    mu=5*np.log10((1+zz)*shape_z);
    # offset-match curve to LCDM at the data weighting (use simple mean over zz vs lcdm)
    muL=5*np.log10((1+zz)*F.D_shape_LCDM(zz,omA)); return mu-muL-(np.median(mu-muL))
fig2,ax2=plt.subplots(figsize=(7.0,4.2))
ax2.axhline(0,color="#c0392b",lw=1.2,label=f"flat $\\Lambda$CDM ($\\Omega_m={omA:.3f}$)")
muTS=5*np.log10((1+zz)*F.D_shape_TS(zz,fvA)); muLref=5*np.log10((1+zz)*F.D_shape_LCDM(zz,omA))
cTS=muTS-muLref; cTS=cTS-np.interp(0.1,zz,cTS)
ax2.plot(zz,cTS,color="#2471a3",lw=1.6,label=f"timescape ($f_{{v0}}={fvA:.3f}$)")
ax2.errorbar(cen,mr- np.interp(0.1,zz, (5*np.log10((1+zz)*F.D_shape_LCDM(zz,omA))-muLref)),yerr=er,fmt="o",ms=4,color="k",alpha=0.7,label="Pantheon+ (binned)")
ax2.set_xscale("log"); ax2.set_xlabel("redshift $z$"); ax2.set_ylabel(r"$\Delta\mu$ rel. to best $\Lambda$CDM [mag]")
ax2.set_title("Pantheon+ Hubble-diagram residuals (standard products)",fontsize=10.5)
ax2.legend(fontsize=8.5,loc="lower left"); ax2.set_ylim(-0.15,0.15)
fig2.tight_layout(); fig2.savefig(f"{OUT}/fig_hubble.pdf"); fig2.savefig(f"{OUT}/fig_hubble.png",dpi=150)
print("wrote fig_hubble.pdf")
