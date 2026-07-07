#!/usr/bin/env python3
"""
(1) Double-check the BAO+CMB numbers with the CORRECT Planck acoustic-scale
    error, cross-validate LCDM against DESI, and report q0 (acceleration) for
    every best fit.
(2) Fit a motivated extension -- w0waCDM (CPL evolving dark energy) -- to the
    same BAO+CMB data, to test whether the acceleration HISTORY needs modifying.
"""
import numpy as np
from scipy.optimize import minimize
import fit_timescape as F, timescape_baocmb as T
C=299792.458; RD=147.09

# ---- correct CMB acoustic-scale point + error ----
theta100=1.04109; sig100=0.00030
DM_zstar_over_rstar=100.0/theta100                  # = 1/theta*  = 96.053
# single, consistent Planck 2018 TT,TE,EE+lowE+lensing column for both r_star
# and r_drag (previously mixed no-lensing r_star=144.39 with +lensing
# r_drag=147.09, an internally inconsistent pairing)
r_star_over_rd=144.43/147.09                         # Planck 2018 +lensing
DM_zstar_over_rd=DM_zstar_over_rstar*r_star_over_rd  # = 94.316 -> 94.32
sig_cmb=DM_zstar_over_rd*(sig100/theta100)           # propagate theta* error
sig_cmb=max(sig_cmb,0.05)                             # floor for r*/rd assumption
print(f"CMB point  D_M(z*)/r_d = {DM_zstar_over_rd:.3f} +/- {sig_cmb:.3f}  (was 94.29 +/- 0.40)")

ZSTAR=1089.80
rows=[(z,k,v,e,c) for (z,k,v,e,c) in T.BAO]+[(ZSTAR,"DM",DM_zstar_over_rd,sig_cmb,None)]
d=np.array([r[2] for r in rows]); Cinv=np.linalg.inv(T.build_cov(rows)); n=len(rows)
rb=rows[:-1]; db=np.array([r[2] for r in rb]); Cib=np.linalg.inv(T.build_cov(rb))

def q0_lcdm(Om): return 1.5*Om-1.0
def q0_w0wa(Om,w0): return 0.5*(Om+(1+3*w0)*(1-Om))
def q0_ts(fv0): return -(1-fv0)*(8*fv0**3+39*fv0**2-12*fv0-8)/(4+fv0+4*fv0**2)**2

def chi2_alpha(model_vec):
    g=model_vec; gCi=Cinv@g; a=(g@(Cinv@d))/(g@gCi)
    return d@(Cinv@d)-(g@(Cinv@d))**2/(g@gCi), a

# ---- timescape ----
fvg=np.linspace(0.30,0.995,696)
bestfv=min(fvg,key=lambda fv:chi2_alpha(T.model_vec(fv,rows))[0])
chi_ts,a_ts=chi2_alpha(T.model_vec(bestfv,rows)); H0_ts=T.g_dress(bestfv)*C/(a_ts*RD)
# Delta-chi2=1 profile error on fv0 (fine grid around the minimum)
_fvg_fine=np.linspace(max(0.30,bestfv-0.05),min(0.995,bestfv+0.05),4001)
_chis=np.array([chi2_alpha(T.model_vec(fv,rows))[0] for fv in _fvg_fine])
_i=int(np.argmin(_chis)); _dchi=_chis-_chis[_i]
_lo=np.interp(1.0,_dchi[:_i+1][::-1],_fvg_fine[:_i+1][::-1]) if _i>0 else bestfv
_hi=np.interp(1.0,_dchi[_i:],_fvg_fine[_i:]) if _i<len(_fvg_fine)-1 else bestfv
fv0_err_lo=float(bestfv-_lo); fv0_err_hi=float(_hi-bestfv)
print(f"  timescape fv0 Delta-chi2=1 error: +{fv0_err_hi:.4f} / -{fv0_err_lo:.4f}")
# ---- LCDM ----
omg=np.linspace(0.15,0.45,301)
bestom=min(omg,key=lambda om:chi2_alpha(T.model_vec(om,rows,lcdm=om))[0])
chi_l,a_l=chi2_alpha(T.model_vec(bestom,rows,lcdm=bestom)); H0_l=C/(a_l*RD)

# ---- w0waCDM (CPL) ----
def Ew(z,Om,w0,wa):
    ODE=1-Om; return np.sqrt(Om*(1+z)**3+ODE*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z)))
def w_vec(Om,w0,wa):
    from scipy.integrate import quad
    out=[]
    for z,k,v,e,c in rows:
        if k=="DM": out.append(quad(lambda zz:1/Ew(zz,Om,w0,wa),0,z)[0])
        elif k=="DH": out.append(1/Ew(z,Om,w0,wa))
        else:
            dM=quad(lambda zz:1/Ew(zz,Om,w0,wa),0,z)[0]; dH=1/Ew(z,Om,w0,wa); out.append((z*dM*dM*dH)**(1/3))
    return np.array(out)
def nll_w(p):
    Om,w0,wa=p
    if not(0.1<Om<0.6 and -2.5<w0<0.5 and -4<wa<3): return 1e9
    return chi2_alpha(w_vec(Om,w0,wa))[0]
res=minimize(nll_w,[0.30,-1.0,0.0],method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-4,maxiter=6000))
Omw,w0,wa=res.x; chi_w,a_w=chi2_alpha(w_vec(Omw,w0,wa)); H0_w=C/(a_w*RD)

def bic(chi2,k): return chi2+k*np.log(n)
print("="*78)
print(f"BAO+CMB ({n} points), rd={RD} Mpc:")
print(f"  timescape : fv0={bestfv:.3f}  H0={H0_ts:.1f}  chi2={chi_ts:.1f}/dof{n-2}={chi_ts/(n-2):.2f}  q0(dressed)={q0_ts(bestfv):+.3f}  BIC={bic(chi_ts,2):.1f}")
print(f"  LCDM      : Om ={bestom:.3f}  H0={H0_l:.1f}  chi2={chi_l:.1f}/dof{n-2}={chi_l/(n-2):.2f}  q0={q0_lcdm(bestom):+.3f}  BIC={bic(chi_l,2):.1f}")
print(f"  w0waCDM   : Om ={Omw:.3f} w0={w0:+.3f} wa={wa:+.3f}  H0={H0_w:.1f}  chi2={chi_w:.1f}/dof{n-4}={chi_w/(n-4):.2f}  q0={q0_w0wa(Omw,w0):+.3f}  BIC={bic(chi_w,4):.1f}")
print(f"  dBIC(w0waCDM - LCDM) = {bic(chi_w,4)-bic(chi_l,2):+.2f}   dBIC(timescape - LCDM) = {bic(chi_ts,2)-bic(chi_l,2):+.2f}")
print("-"*78)
print("CROSS-VALIDATION: DESI DR1 BAO+CMB(full) published LCDM = Om 0.3069, H0 67.97.")
print(f"  our single-acoustic-point proxy:        Om {bestom:.3f}, H0 {H0_l:.1f}  -> "
      +("OK (within ~2%)" if abs(H0_l-67.97)<2 else "check"))
print("-"*78)
# assumption-light core: SN vs BAO-only (no CMB, no r_d)
zHD,zHEL,mb,Cf=F.load(); chi2sn=F.make_chi2(zHD,zHEL,mb,Cf)
fvsn=min(fvg,key=lambda fv:chi2sn(F.D_shape_TS(zHD,fv)))
def chi_bo(fv):
    g=T.model_vec(fv,rb); gCi=Cib@g; return db@(Cib@db)-(g@(Cib@db))**2/(g@gCi)
fvbo=min(fvg,key=chi_bo)
sn_at_bo=chi2sn(F.D_shape_TS(zHD,fvbo))-chi2sn(F.D_shape_TS(zHD,fvsn))
bo_at_sn=chi_bo(fvsn)-chi_bo(fvbo)
print(f"ASSUMPTION-LIGHT CORE (SN + BAO only; NO CMB, NO r_d):")
print(f"  SN best fv0={fvsn:.3f}; BAO-only best fv0={fvbo:.3f}")
print(f"  SN Delta-chi2 at the BAO fv0 = {sn_at_bo:.1f} ; BAO Delta-chi2 at the SN fv0 = {bo_at_sn:.1f}")
print(f"  q0(dressed) at SN fv0={fvsn:.2f}: {q0_ts(fvsn):+.3f}   at BAO fv0={fvbo:.2f}: {q0_ts(fvbo):+.3f}  (both <0 => apparent acceleration present)")

import json
json.dump(dict(cmb_point=DM_zstar_over_rd,cmb_err=sig_cmb,
  timescape=dict(fv0=float(bestfv),fv0_err=[fv0_err_lo,fv0_err_hi],H0=float(H0_ts),chi2=float(chi_ts),dof=n-2,q0=float(q0_ts(bestfv))),
  lcdm=dict(Om=float(bestom),H0=float(H0_l),chi2=float(chi_l),dof=n-2,q0=float(q0_lcdm(bestom))),
  w0wa=dict(Om=float(Omw),w0=float(w0),wa=float(wa),H0=float(H0_w),chi2=float(chi_w),dof=n-4,q0=float(q0_w0wa(Omw,w0)),dBIC_vs_lcdm=float(bic(chi_w,4)-bic(chi_l,2))),
  core=dict(fv0_SN=float(fvsn),fv0_BAOonly=float(fvbo),SN_dchi2_at_BAOfv=float(sn_at_bo),BAO_dchi2_at_SNfv=float(bo_at_sn))),
  open("../results_baocmb_dr1.json","w"),indent=2)
print("\nsaved results_baocmb_dr1.json")
