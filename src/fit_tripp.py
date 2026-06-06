#!/usr/bin/env python3
"""
Cosmology-INDEPENDENT Tripp reduction of Pantheon+ (à la Lane/Seifert 2024):
fit raw SALT2 parameters (mB, x1, c) with the Tripp relation and free
(alpha, beta, sigma_int), NO BBC bias correction (which assumes an FLRW
fiducial), and compare timescape vs flat LCDM.

    mu_Tripp = mB + alpha*x1 - beta*c - M
    Var(mu) = mBERR^2 + alpha^2 x1ERR^2 + beta^2 cERR^2
              + 2 alpha cov(mB,x1) - 2 beta cov(mB,c) - 2 alpha beta cov(x1,c)
              + vpec_err^2 + sigma_int^2
    cov(mB,x1) = -k cov(x0,x1),  cov(mB,c) = -k cov(x0,c),  k = 2.5/(x0 ln10)

For each cosmology value we maximise the Gaussian likelihood over
(alpha, beta, sigma_int) with M marginalised analytically, then compare the
two models by Delta(-2 lnL_max) (= Delta BIC, equal nuisance count).
Statistical (diagonal) errors only -- the cosmology-dependent systematic
covariance is deliberately excluded.
"""
import numpy as np
from scipy.optimize import minimize
import fit_timescape as F

LN10 = np.log(10.0)

def load_salt():
    with open(F.DATA) as f:
        header = f.readline().split()
        idx = {n:i for i,n in enumerate(header)}
        rows = [ln.split() for ln in f]
    def col(name): return np.array([float(r[idx[name]]) for r in rows])
    d = {n: col(n) for n in
         ["zHD","zHEL","mB","mBERR","x1","x1ERR","c","cERR","x0","x0ERR",
          "COV_x1_c","COV_x1_x0","COV_c_x0","m_b_corr_err_VPEC"]}
    iscal = np.array([int(float(r[idx["IS_CALIBRATOR"]])) for r in rows])
    m = (iscal==0) & (d["zHD"]>0.01)
    return {k:v[m] for k,v in d.items()}

def neg2lnL_factory(D, mu_shape):
    mB,x1,c = D["mB"],D["x1"],D["c"]
    vmB = D["mBERR"]**2; vx1=D["x1ERR"]**2; vc=D["cERR"]**2
    k = 2.5/(D["x0"]*LN10)
    cov_mB_x1 = -k*D["COV_x1_x0"]
    cov_mB_c  = -k*D["COV_c_x0"]
    cov_x1_c  = D["COV_x1_c"]
    vpec = D["m_b_corr_err_VPEC"]**2
    def n2(p):
        a,b,ls = p; sig2 = np.exp(2*ls)
        var_diag = vmB + a*a*vx1 + b*b*vc + vpec + sig2          # always > 0
        var = (var_diag + 2*a*cov_mB_x1 - 2*b*cov_mB_c - 2*a*b*cov_x1_c)
        # fall back to diagonal where the linearized cross-cov makes var<=0
        # (a few faint SNe with tiny x0; the cross terms are unreliable there)
        var = np.where(var > 0, var, var_diag)
        w = 1.0/var
        r0 = (mB + a*x1 - b*c) - mu_shape
        M = np.sum(w*r0)/np.sum(w)
        r = r0 - M
        return np.sum(r*r*w) + np.sum(np.log(2*np.pi*var))
    return n2

def best(D, mu_shape):
    n2 = neg2lnL_factory(D, mu_shape)
    res = minimize(n2, x0=[0.14,3.1,np.log(0.1)], method="Nelder-Mead",
                   options=dict(xatol=1e-5,fatol=1e-5,maxiter=4000))
    return res.fun, res.x

def scan_model(D, shape_fn, grid):
    vals=[]; pars=[]
    for g in grid:
        mu = 5*np.log10((1+D["zHEL"])*shape_fn(g))
        f,p = best(D, mu); vals.append(f); pars.append(p)
    vals=np.array(vals); i=int(np.argmin(vals))
    dd=vals-vals[i]
    lo = np.interp(1.0, dd[:i+1][::-1], grid[:i+1][::-1]) if i>0 else grid[i]
    hi = np.interp(1.0, dd[i:], grid[i:]) if i<len(grid)-1 else grid[i]
    return grid[i], vals[i], grid[i]-lo, hi-grid[i], pars[i], grid, vals

def run(D, tag):
    fvg=np.linspace(0.50,0.90,81); omg=np.linspace(0.05,0.60,81)
    fv,LTS,a1,a2,pTS,_,_ = scan_model(D, lambda fv:F.D_shape_TS(D["zHD"],fv), fvg)
    om,LL ,o1,o2,pL ,_,_ = scan_model(D, lambda om:F.D_shape_LCDM(D["zHD"],om), omg)
    print(f"{tag}: N={len(D['zHD'])}")
    print(f"  TIMESCAPE fv0 = {fv:.3f} (+{a2:.3f}/-{a1:.3f})  alpha={pTS[0]:.3f} beta={pTS[1]:.3f} "
          f"sig_int={np.exp(pTS[2]):.3f}  -2lnL={LTS:.2f}")
    print(f"  LCDM      Om0 = {om:.3f} (+{o2:.3f}/-{o1:.3f})  alpha={pL[0]:.3f} beta={pL[1]:.3f} "
          f"sig_int={np.exp(pL[2]):.3f}  -2lnL={LL:.2f}")
    print(f"  Delta(-2lnL) = dBIC (TS - LCDM) = {LTS-LL:+.2f}   (negative favours timescape)")
    return fv,om,LTS,LL

if __name__=="__main__":
    D = load_salt()
    print("="*70)
    print("COSMOLOGY-INDEPENDENT TRIPP REDUCTION (stat-only)")
    print("="*70)
    run(D, "Full sample (zHD>0.01)")
    print()
    for zc in [0.023,0.075]:
        m = D["zHD"]>zc
        run({k:v[m] for k,v in D.items()}, f"zHD>{zc}")
        print()
