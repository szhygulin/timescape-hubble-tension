#!/usr/bin/env python3
"""
Proper joint SN + BAO + CMB fit. Tests the motivated extension w0waCDM (CPL
evolving dark energy) against LCDM, and shows timescape cannot be fit jointly.

Total chi2(cosmology) = chi2_SN(shape; offset marginalised)
                      + chi2_BAO+CMB(distances; alpha=c/(H0 rd) marginalised).
Shared cosmological parameters; separate nuisances per dataset.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid as cumtrap
import fit_timescape as F, timescape_baocmb as T
C=299792.458; RD=147.09

# ---- SN data + full covariance ----
zHD,zHEL,mb,Cf=F.load(); SN_chi2=F.make_chi2(zHD,zHEL,mb,Cf)
# ---- BAO+CMB data (correct CMB error) ----
DMz=(100.0/1.04109)*(144.43/147.09); sig=max(DMz*(0.00030/1.04109),0.05)  # Planck 2018 +lensing column (consistent r_star/r_drag; was mixed 144.39/147.09)
rows=[(z,k,v,e,c) for (z,k,v,e,c) in T.BAO]+[(1089.80,"DM",DMz,sig,None)]
d=np.array([r[2] for r in rows]); Cinv=np.linalg.inv(T.build_cov(rows)); nB=len(rows)
def bc_chi2(vec):
    g=vec; gCi=Cinv@g; a=(g@(Cinv@d))/(g@gCi); return d@(Cinv@d)-(g@(Cinv@d))**2/(g@gCi), a

# ---- E(z) and comoving-distance shapes ----
def Ew(z,Om,w0,wa):
    ODE=1-Om; return np.sqrt(Om*(1+z)**3+ODE*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z)))
_zg=np.linspace(0,zHD.max()*1.0001,400000)
def Dc_SN(Om,w0,wa):
    invE=1/Ew(_zg,Om,w0,wa); Dc=cumtrap(invE,_zg,initial=0); return np.interp(zHD,_zg,Dc)
def bc_vec_w(Om,w0,wa):
    from scipy.integrate import quad
    out=[]
    for z,k,v,e,c in rows:
        if k=="DM": out.append(quad(lambda zz:1/Ew(zz,Om,w0,wa),0,z)[0])
        elif k=="DH": out.append(1/Ew(z,Om,w0,wa))
        else:
            dM=quad(lambda zz:1/Ew(zz,Om,w0,wa),0,z)[0]; dH=1/Ew(z,Om,w0,wa); out.append((z*dM*dM*dH)**(1/3))
    return np.array(out)

def joint_w(p):
    Om,w0,wa=p
    if not(0.1<Om<0.6 and -2.5<w0<0.5 and -4<wa<2): return 1e9
    csn=SN_chi2(Dc_SN(Om,w0,wa)); cbc,_=bc_chi2(bc_vec_w(Om,w0,wa)); return csn+cbc
def joint_lcdm(Om):
    csn=SN_chi2(Dc_SN(Om,-1,0)); cbc,_=bc_chi2(bc_vec_w(Om,-1,0)); return csn+cbc

N=len(zHD)+nB
# LCDM joint
from scipy.optimize import minimize_scalar
rl=minimize_scalar(joint_lcdm,bounds=(0.15,0.45),method="bounded")
Oml=rl.x; chil=rl.fun; _,al=bc_chi2(bc_vec_w(Oml,-1,0)); H0l=C/(al*RD)
# w0waCDM joint
rw=minimize(joint_w,[0.30,-0.9,-0.1],method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-4,maxiter=8000))
Omw,w0,wa=rw.x; chiw=rw.fun; _,aw=bc_chi2(bc_vec_w(Omw,w0,wa)); H0w=C/(aw*RD)
# timescape joint (single fv0 for SN+BAO+CMB)
fvg=np.linspace(0.40,0.95,400)
def joint_ts(fv): return SN_chi2(F.D_shape_TS(zHD,fv))+bc_chi2(T.model_vec(fv,rows))[0]
fvj=min(fvg,key=joint_ts); chits=joint_ts(fvj); _,ats=bc_chi2(T.model_vec(fvj,rows)); H0ts=T.g_dress(fvj)*C/(ats*RD)

def bic(chi2,k): return chi2+k*np.log(N)
# nuisances: SN offset (1) + BAO alpha (1) = 2 shared; +cosmology
kL=2+1; kW=2+3; kT=2+1
print("="*82)
print(f"JOINT SN + BAO + CMB  (N={N}: {len(zHD)} SNe + {nB} BAO/CMB)")
print("="*82)
print(f"LCDM     : Om={Oml:.3f}                         H0={H0l:.1f}  chi2={chil:.1f}  BIC={bic(chil,kL):.1f}")
print(f"w0waCDM  : Om={Omw:.3f} w0={w0:+.3f} wa={wa:+.3f}  H0={H0w:.1f}  chi2={chiw:.1f}  BIC={bic(chiw,kW):.1f}  q0={0.5*(Omw+(1+3*w0)*(1-Omw)):+.3f}")
print(f"timescape: fv0={fvj:.3f}                        H0={H0ts:.1f}  chi2={chits:.1f}  BIC={bic(chits,kT):.1f}")
print("-"*82)
print(f"dBIC(w0waCDM - LCDM)  = {bic(chiw,kW)-bic(chil,kL):+.2f}   (negative => evolving DE preferred)")
print(f"dBIC(timescape - LCDM)= {bic(chits,kT)-bic(chil,kL):+.2f}   (positive => LCDM preferred)")
print(f"Dchi2(w0waCDM vs LCDM)= {chil-chiw:+.1f} for 2 extra params")
import json
json.dump(dict(N=N,
  lcdm=dict(Om=float(Oml),H0=float(H0l),chi2=float(chil)),
  w0wa=dict(Om=float(Omw),w0=float(w0),wa=float(wa),H0=float(H0w),chi2=float(chiw),
            dBIC_vs_lcdm=float(bic(chiw,kW)-bic(chil,kL)),dchi2_vs_lcdm=float(chil-chiw)),
  timescape=dict(fv0=float(fvj),H0=float(H0ts),chi2=float(chits),dBIC_vs_lcdm=float(bic(chits,kT)-bic(chil,kL)))),
  open("../results_joint.json","w"),indent=2)
print("saved results_joint.json")
