#!/usr/bin/env python3
"""Figure: the void fraction preferred by supernovae vs by BAO+CMB are
incompatible -> timescape cannot fit both with one fv0."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fit_timescape as F, timescape_baocmb as T
OUT=".."

# SN chi2(fv0), full covariance, standard m_b_corr
zHD,zHEL,mb,Cf=F.load(); chi2_sn=F.make_chi2(zHD,zHEL,mb,Cf)
fvg=np.linspace(0.40,0.95,300)
sn=np.array([chi2_sn(F.D_shape_TS(zHD,fv)) for fv in fvg]); sn-=sn.min()
# BAO+CMB chi2(fv0)
rows=T.build_data(); d=np.array([r[2] for r in rows]); Ci=np.linalg.inv(T.build_cov(rows))
bc=np.array([T.fit([fv],rows,Ci,d)[2] for fv in fvg]); bc-=bc.min()
# BAO-only
rb=rows[:-1]; db=np.array([r[2] for r in rb]); Cib=np.linalg.inv(T.build_cov(rb))
bo=np.array([T.fit([fv],rb,Cib,db)[2] for fv in fvg]); bo-=bo.min()

fvsn=fvg[np.argmin(sn)]; fvbc=fvg[np.argmin(bc)]
json.dump({"fv0_SN":float(fvsn),"fv0_BAOCMB":float(fvbc),"fv0_BAO":float(fvg[np.argmin(bo)])},
          open(f"{OUT}/results_fvtension.json","w"),indent=2)

fig,ax=plt.subplots(figsize=(7.0,4.3))
ax.plot(fvg,sn,color="#2471a3",lw=2,label=f"Supernovae (Pantheon+), min $f_{{v0}}\\approx{fvsn:.2f}$")
ax.plot(fvg,bc,color="#c0392b",lw=2,label=f"BAO + CMB acoustic scale, min $f_{{v0}}\\approx{fvbc:.2f}$")
ax.plot(fvg,bo,color="#e67e22",lw=1.3,ls="--",label=f"BAO only, min $f_{{v0}}\\approx{fvg[np.argmin(bo)]:.2f}$")
for y in (1,4,9): ax.axhline(y,color="grey",ls=":",lw=0.6)
ax.text(0.405,1.1,"$1\\sigma$",fontsize=8,color="grey"); ax.text(0.405,4.1,"$2\\sigma$",fontsize=8,color="grey"); ax.text(0.405,9.1,"$3\\sigma$",fontsize=8,color="grey")
ax.set_xlabel("present void fraction $f_{v0}$"); ax.set_ylabel(r"$\Delta\chi^2$")
ax.set_ylim(0,25); ax.set_xlim(0.40,0.95)
ax.set_title("Timescape: supernovae and BAO+CMB require incompatible void fractions",fontsize=10)
ax.legend(fontsize=8.5,loc="upper center")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_fvtension.pdf"); fig.savefig(f"{OUT}/fig_fvtension.png",dpi=150)
print("fv0_SN=%.3f  fv0_BAO+CMB=%.3f  fv0_BAO=%.3f"%(fvsn,fvbc,fvg[np.argmin(bo)]))
print("wrote fig_fvtension.pdf")
PY=0
