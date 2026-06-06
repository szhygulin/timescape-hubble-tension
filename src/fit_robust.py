#!/usr/bin/env python3
"""
Driver isolation: hold the FULL Pantheon+ STAT+SYS covariance fixed and swap
ONLY the apparent-magnitude definition, to pin what flips the timescape-vs-LCDM
verdict.

  (1) standard      : m_b_corr                      (BBC bias-corrected; cosmology-dependent)
  (2) bias removed  : m_b_corr + biasCor_m_b        (undo the FLRW-fiducial bias correction)
  (3) raw Tripp     : mB + a0*x1 - b0*c             (own standardisation, no bias corr)

All three use the same covariance C_full and the same offset marginalisation,
so any change in dBIC is due solely to the magnitude standardisation.
"""
import numpy as np
import fit_timescape as F

A0, B0 = 0.131, 2.663   # from the diagonal Tripp fit (full sample)

def load_mags():
    with open(F.DATA) as f:
        header=f.readline().split(); idx={n:i for i,n in enumerate(header)}
        rows=[ln.split() for ln in f]
    col=lambda n: np.array([float(r[idx[n]]) for r in rows])
    d={n:col(n) for n in ["zHD","zHEL","m_b_corr","mB","x1","c","biasCor_m_b"]}
    iscal=np.array([int(float(r[idx["IS_CALIBRATOR"]])) for r in rows])
    m=(iscal==0)&(d["zHD"]>0.01)
    with open(F.COV) as f: n=int(f.readline())
    C=np.loadtxt(F.COV,skiprows=1).reshape(n,n)[np.ix_(m,m)]
    return {k:v[m] for k,v in d.items()}, C

def scan_mag(mag, zHD, zHEL, C, tag):
    chi2=F.make_chi2_mag(mag,zHEL,C) if hasattr(F,"make_chi2_mag") else None
    # inline chi2 with given magnitude vector
    Cinv=np.linalg.inv(C); one=np.ones_like(mag); s11=one@(Cinv@one)
    def chi2(D_shape):
        r=mag-5*np.log10((1+zHEL)*D_shape); Cr=Cinv@r
        return r@Cr-(one@Cr)**2/s11
    fvg=np.linspace(0.50,0.90,400); omg=np.linspace(0.05,0.60,300)
    fv,cTS,a,b,_,_=F.scan(chi2,lambda fv:F.D_shape_TS(zHD,fv),fvg,"")
    om,cL ,c,d,_,_=F.scan(chi2,lambda om:F.D_shape_LCDM(zHD,om),omg,"")
    print(f"{tag:34s} fv0={fv:.3f}  Om={om:.3f}  chi2_TS={cTS:7.1f} chi2_L={cL:7.1f}  dBIC={cTS-cL:+6.2f}  "
          + ("timescape" if cTS<cL else "LCDM"))
    return cTS-cL

D,C=load_mags()
# sanity: confirm m_b_corr ~ mB + a x1 - b c - biasCor for Pantheon+ a~0.15,b~3.1
approx = D["mB"]+0.15*D["x1"]-3.1*D["c"]-D["biasCor_m_b"]
print(f"# convention check: median |m_b_corr - (mB+0.15x1-3.1c-biasCor)| = "
      f"{np.median(np.abs(D['m_b_corr']-approx)):.3f} mag (small => bias subtracted)")
print(f"# biasCor_m_b: median={np.median(D['biasCor_m_b']):+.4f}  range [{D['biasCor_m_b'].min():+.3f},{D['biasCor_m_b'].max():+.3f}]")
print("="*92)
print("SAME full STAT+SYS covariance; only the magnitude definition changes:")
scan_mag(D["m_b_corr"],                         D["zHD"],D["zHEL"],C,"(1) standard m_b_corr")
scan_mag(D["m_b_corr"]+D["biasCor_m_b"],        D["zHD"],D["zHEL"],C,"(2) bias correction removed")
scan_mag(D["mB"]+A0*D["x1"]-B0*D["c"],          D["zHD"],D["zHEL"],C,"(3) raw Tripp (own a,b)")
