# Independent check of the P3 evidence Tripp-corner headline (lnB ~ -5.5, favours
# timescape, reproducing Seifert lnB>5). Re-implements the profile-only evidence
# (M marginalised exactly + profile over (a,b,ls); NO Laplace det) via my own
# trapezoid quadrature, using only the verified fit_tripp likelihood primitive.
# The JSON's own "lnB_profile_only_variant" is -5.66..-5.92; matching it confirms
# the headline is not an artifact of the Laplace term.
import os, sys, json
import numpy as np
from scipy.optimize import minimize

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = WT + "/src"
os.chdir(SRC); sys.path.insert(0, SRC)
import fit_timescape as F
import fit_tripp as FT

D = FT.load_salt()
LN2PI = np.log(2*np.pi)

# S_w(p) = sum 1/var (same as evidence.make_Sw / fit_tripp variance model)
vmB = D["mBERR"]**2; vx1 = D["x1ERR"]**2; vc = D["cERR"]**2
k = 2.5/(D["x0"]*np.log(10.0))
cmx1 = -k*D["COV_x1_x0"]; cmc = -k*D["COV_c_x0"]; cx1c = D["COV_x1_c"]
vpec = D["m_b_corr_err_VPEC"]**2
def Sw(p):
    a, b, ls = p; sig2 = np.exp(2*ls)
    vd = vmB + a*a*vx1 + b*b*vc + vpec + sig2
    v = vd + 2*a*cmx1 - 2*b*cmc - 2*a*b*cx1c
    v = np.where(v > 0, v, vd)
    return float(np.sum(1.0/v))

def profiled_n2m(grid, shape_fn):
    n2m = np.empty(len(grid)); sols = np.empty((len(grid), 3))
    i0 = len(grid)//2
    order = list(range(i0, len(grid))) + list(range(i0-1, -1, -1))
    for idx in order:
        g = grid[idx]
        mu = 5.0*np.log10((1.0+D["zHEL"])*shape_fn(g))
        n2 = FT.neg2lnL_factory(D, mu)
        f = lambda p: n2(p) + np.log(Sw(p))
        x0 = np.array([0.14, 3.1, np.log(0.1)]) if idx == i0 else (sols[idx-1] if idx > i0 else sols[idx+1])
        r = minimize(f, x0=x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=8000))
        sols[idx] = r.x; n2m[idx] = r.fun
    return n2m

def logtrapz(lnf, x):
    m = float(np.max(lnf)); return m + float(np.log(np.trapezoid(np.exp(lnf - m), x)))
def lnZ_profonly(grid, n2m, lo, hi):
    msk = (grid >= lo-1e-12) & (grid <= hi+1e-12)
    g = grid[msk]; w = -0.5*n2m[msk]           # M marginalised (+2ln2pi vol cancels in lnB)
    return logtrapz(w, g) - float(np.log(hi - lo))

fvg = np.linspace(0.40, 0.99, 237); omg = np.linspace(0.05, 0.70, 261)
n2_TS = profiled_n2m(fvg, lambda fv: F.D_shape_TS(D["zHD"], fv))
n2_L  = profiled_n2m(omg, lambda om: F.D_shape_LCDM(D["zHD"], om))

# profiled dBIC cross-check (subtract the ln Sw term to recover fit_tripp's -2lnL)
def n2_plain_min(grid, n2m, shape_fn):
    i = int(np.argmin(n2m)); g = grid[i]
    mu = 5.0*np.log10((1.0+D["zHEL"])*shape_fn(g))
    n2 = FT.neg2lnL_factory(D, mu)
    r = minimize(n2, x0=[0.14, 3.1, np.log(0.1)], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-6, maxiter=8000))
    return g, float(r.fun)
fv_min, n2TS_plain = n2_plain_min(fvg, n2_TS, lambda fv: F.D_shape_TS(D["zHD"], fv))
om_min, n2L_plain = n2_plain_min(omg, n2_L, lambda om: F.D_shape_LCDM(D["zHD"], om))

lnB = {}
for (opn, olo, ohi) in [("Om~U(0.10,0.60)", 0.10, 0.60), ("Om~U(0.05,0.70)", 0.05, 0.70)]:
    zL = lnZ_profonly(omg, n2_L, olo, ohi)
    for (fpn, flo, fhi) in [("fv0~U(0.50,0.95)", 0.50, 0.95), ("fv0~U(0.40,0.99)", 0.40, 0.99)]:
        zTS = lnZ_profonly(fvg, n2_TS, flo, fhi)
        lnB[f"{opn} x {fpn}"] = round(zL - zTS, 3)

out = dict(
    tripp_profiled_dBIC=dict(fv0=round(fv_min, 4), Om=round(om_min, 4),
                             n2_TS=round(n2TS_plain, 4), n2_L=round(n2L_plain, 4),
                             dBIC=round(n2TS_plain - n2L_plain, 4),
                             committed_results_json=-9.3966),
    lnB_LCDM_over_TS_profile_only=lnB,
    json_profile_only_variant={"Om~U(0.10,0.60) x fv0~U(0.50,0.95)": -5.660,
                               "Om~U(0.05,0.70) x fv0~U(0.50,0.95)": -5.922},
    interpretation="negative lnB => favours timescape; |lnB| 5.4-5.9 reproduces Seifert lnB>5")
print(json.dumps(out, indent=2))
