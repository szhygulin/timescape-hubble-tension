#!/usr/bin/env python3
"""
Rasanen statistical backreaction / "peak model" (Buchert averaging with a
structure-formation-based evaluation), fit to Pantheon+ SN + DESI BAO + Planck
CMB acoustic scale on the SAME harness footing as the timescape analysis.

PRIMARY LITERATURE
------------------
[R08a] Rasanen 2008, "Evaluating backreaction with the peak model of structure
       formation", JCAP 0804:026  (arXiv:0801.2692).
[R08b] Rasanen 2008, "The effect of structure formation on the expansion of the
       universe" (essay)  (arXiv:0805.2670).
[R09 ] Rasanen 2009, "Light propagation in statistically homogeneous and
       isotropic dust universes", JCAP 0902:011  (arXiv:0812.2872).
  (Note: the task prompt's arXiv:0903.3013 is a DIFFERENT Rasanen 2009 paper on
   CMB isotropy / EGS theorems; the distance-prescription paper is 0812.2872.)

DISTANCE PRESCRIPTION (the key result, [R09])
---------------------------------------------
In a statistically homogeneous and isotropic dust universe, light propagation
is governed by the AVERAGE geometry:
  (za)    1 + z_bar = 1 / a            a = volume-average (Buchert) scale factor
  (DAeqav) H d_zbar[ (1+z_bar)^2 H d_zbar D_A ] = -[4 pi G rho_bar + shear] D_A
Neglecting the null shear (justified in [R09]: shear suppressed over the
homogeneity scale), (DAeqav) has EXACTLY the FRW form driven by H(z_bar) and the
present matter density. [R09] states it explicitly:
  "A clumpy model with the same H(z_bar) (and present-day matter density) as the
   LCDM model has the same average distance-redshift relation."
=> D_M(z) = int_0^z dz'/E(z'),  D_H(z) = 1/E(z),  E = H/H0  (units c/H0),
   with E(z) set by the AVERAGE expansion history. The ONLY model freedom is the
   shape of H(z_bar); the distance map is FRW-standard once H is fixed.

AVERAGE EXPANSION HISTORY (Buchert first integral, [R08a] eq. firstQ)
---------------------------------------------------------------------
  3 H^2 = 8 pi G rho_bar0/a^3 - 3K/a^2 + (2/a^2) int^a da'/a' a'^2 Q
  constraint:  1 = Om + OR + OQ     (Om=8piG rho/3H^2, OR=-R/6H^2, OQ=-Q/6H^2)
  deceleration: q = (1/2) Om + 2 OQ                          [R08a] eq. (q)
The integrability condition fixes the simplest non-trivial closure: when the
average curvature carries an a^-2 piece, the backreaction Q evolves as a^-6
([R08a] after eq. QfirstR). With Q ~ a^-6 and curvature ~ a^-2 the first
integral closes to
  E(z)^2 = Om0 (1+z)^3 + OR0 (1+z)^2 + OQ0 (1+z)^6 ,   Om0+OR0+OQ0 = 1.
This is the as-published Buchert closure; OQ0 is the present backreaction
density parameter (the ONE structure-formation observable the peak model
predicts), OR0 the average-curvature parameter.

PEAK-MODEL PREDICTION (parameter-free, [R08a])
----------------------------------------------
Given the primordial spectrum + CDM transfer function, the peak model FIXES the
expansion with NO free parameters. The published result ([R08a] figs Ht, q,
omegas; abstract):
  * Ht rises from 2/3 to "somewhat less than unity" (Ht <= 1 always).
  * |OQ| < 0.04 at ALL times.
  * q = -addot/(aH^2) > 0 at ALL times -> NO ACCELERATION.
  * |OQ| >~ 0.2 is REQUIRED for acceleration of the observed magnitude; the peak
    model falls short by a factor ~5+.
So the validation target for our E(z) is the published bound OQ0 in [-0.04,0],
q0 > 0. We then ALSO let OQ0 float freely against SN to measure what magnitude
the data DEMAND, and compare to the peak-model 0.04 ceiling.
"""
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid as cumtrap
from scipy.optimize import minimize, minimize_scalar
import harness as H

C_KM = 299792.458

# ---------------------------------------------------------------------------
# Average expansion history E(z) = H/H0 from the Buchert first integral with the
# a^-6 backreaction / a^-2 curvature closure ([R08a] firstQ + integrability).
# Free shape params: (Om0, OQ0); OR0 = 1 - Om0 - OQ0 (flat-total constraint).
# ---------------------------------------------------------------------------
# Buchert first integral ([R08a] firstQ) closes, with the integrability-fixed
# Q ~ a^-6 and curvature ~ a^-2 pieces, to
#   E(z)^2 = Om0 (1+z)^3 + OR0 (1+z)^2 + OQ0 (1+z)^6 ,  OR0 = 1 - Om0 - OQ0.
# The a^-6 backreaction term is the FORMALLY exact late-time closure but, taken
# literally, dominates at the CMB redshift (z~1090) where the universe is in fact
# near-FRW (the peak model has |OQ|<0.04 TODAY and backreaction grows only at
# t~10-100 Gyr, i.e. z<~1). To keep the high-z limit physical (EdS dust recovery,
# as the peak model demands) we present TWO closures:
#   mode="a6"  : literal a^-6 term  (exact Buchert closure; pathological at CMB)
#   mode="late": backreaction confined to late times via a (1+z)^6/(1+z_t)^6
#                saturation that -> 0 at high z, so high-z -> FRW dust+curvature.
# Both share the SAME today-value OQ0 and q0; they differ only in early-time tail.
_ZT = 1.0  # backreaction turn-on redshift (structure formation, t~10 Gyr)

def E2(z, Om0, OQ0, mode="a6"):
    OR0 = 1.0 - Om0 - OQ0
    z = np.asarray(z, dtype=float)
    if mode == "a6":
        # literal Buchert a^-6 closure ([R08a]); exact but blows up at the CMB
        return Om0*(1+z)**3 + OR0*(1+z)**2 + OQ0*(1+z)**6
    # late-time-confined: backreaction tracks structure formation, so it is
    # present only for z<~_ZT and vanishes at high z (EdS/FRW dust recovery, as
    # the peak model requires). The today-value is OQ0; the a^-6 growth is
    # switched off above _ZT with a smooth logistic in a=1/(1+z).
    a = 1.0/(1.0+z); at = 1.0/(1.0+_ZT)
    s = 1.0/(1.0 + (at/np.maximum(a, 1e-30))**6)   # ->1 as a->1, ->0 as a->0
    s0 = 1.0/(1.0 + (at)**6)                        # normalisation at a=1 (z=0)
    return Om0*(1+z)**3 + OR0*(1+z)**2 + OQ0*(1+z)**6*(s/s0)

def invE(z, Om0, OQ0, mode="a6"):
    e2 = E2(z, Om0, OQ0, mode)
    e2 = np.where(np.asarray(e2) > 1e-12, e2, 1e-12)  # guard unphysical corners
    return 1.0/np.sqrt(e2)

def q0_of(Om0, OQ0):
    # q = 1/2 Om + 2 OQ  ([R08a] eq. q), evaluated today (Om(today)=Om0 etc.)
    return 0.5*Om0 + 2.0*OQ0

# shared comoving-distance grid to the SN max redshift (~2.3); built once
_zHD_max = float(np.max(H.load_sn()[0]))
_ZG_SN = np.linspace(0.0, _zHD_max*1.0001, 60000)
# log-spaced grid to z* for the CMB / BAO predict (captures z<1 and z~1090)
_ZG_CMB = np.unique(np.concatenate([
    np.linspace(0.0, 5.0, 5000), np.linspace(5.0, 1100.0, 6000)]))

def Dc(zarr, Om0, OQ0, mode="a6"):
    dc = cumtrap(invE(_ZG_SN, Om0, OQ0, mode), _ZG_SN, initial=0.0)
    return np.interp(zarr, _ZG_SN, dc)

def predict(Om0, OQ0, mode="a6"):
    dcM = cumtrap(invE(_ZG_CMB, Om0, OQ0, mode), _ZG_CMB, initial=0.0)
    def p(z, kind):
        ee = float(invE(np.array([z]), Om0, OQ0, mode)[0])
        if kind == "DH":
            return ee
        dM = float(np.interp(z, _ZG_CMB, dcM))
        if kind == "DM":
            return dM
        return (z*dM*dM*ee)**(1/3)
    return p

# ---------------------------------------------------------------------------
# VALIDATION of the distance code against a published number.
# [R09]: the averaged distance ODE reduces EXACTLY to the FRW form when shear is
# neglected. Hence for OQ0=0 (no backreaction) E(z) -> a flat/open dust+curvature
# FRW model and our D_M must coincide with the harness's own LCDM machinery for
# the matching FRW limit. We check our invE/Dc against an independent quad of the
# SAME integrand, and against the harness LCDM Dc in the pure-dust flat limit
# (Om0=1, OQ0=0 => E^2=(1+z)^3, the Einstein-de Sitter distance d_C analytic
#  value = 2[1 - 1/sqrt(1+z)]).
# ---------------------------------------------------------------------------
def validate():
    msgs = []
    # (1) EdS closed form: Om0=1, OQ0=0 -> E=(1+z)^{3/2}, Dc=2(1-1/sqrt(1+z))
    z = np.array([0.1, 0.5, 1.0, 2.0])
    dc_num = Dc(z, 1.0, 0.0)
    dc_ana = 2.0*(1.0 - 1.0/np.sqrt(1.0+z))
    err = np.max(np.abs(dc_num - dc_ana))
    msgs.append(f"EdS Dc max|num-analytic|={err:.2e} (z={list(z)})")
    assert err < 1e-4, "EdS distance check failed"
    # (2) cross-check against harness LCDM in the OQ0=0, OR0=0 flat-LCDM limit:
    #     E^2=Om(1+z)^3+(1-Om) is NOT reachable here (our curvature term is
    #     (1+z)^2 not constant), but for Om0=1 both reduce to EdS -> already (1).
    # (3) predict() DH vs 1/sqrt(E2) and DM vs cumulative integral consistency:
    p = predict(0.3, -0.02)
    dh = p(0.5, "DH"); dh_ref = 1.0/np.sqrt(float(E2(0.5, 0.3, -0.02)))
    msgs.append(f"DH consistency |p-ref|={abs(dh-dh_ref):.2e}")
    assert abs(dh-dh_ref) < 1e-9
    # (4) predict DM vs Dc grid agreement at a survey redshift
    dm_p = predict(0.3, -0.02)(1.0, "DM"); dm_g = float(Dc(np.array([1.0]),0.3,-0.02)[0])
    msgs.append(f"DM(predict) vs Dc(grid) |d|={abs(dm_p-dm_g):.2e}")
    assert abs(dm_p-dm_g) < 1e-3
    return "; ".join(msgs)

# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fit_sn_only(om_grid, oq_grid, mode="a6"):
    zHD, *_ = H.load_sn()
    best = None
    for om in om_grid:
        for oq in oq_grid:
            c = H.sn_chi2(Dc(zHD, om, oq, mode))
            if best is None or c < best[2]:
                best = (om, oq, c)
    return best

def fit_baocmb(om_grid, oq_grid, cmb=True, mode="a6"):
    chi = H.bao_cmb_chi2 if cmb else H.bao_only_chi2
    best = None
    for om in om_grid:
        for oq in oq_grid:
            c, a = chi(predict(om, oq, mode))
            if best is None or c < best[2]:
                best = (om, oq, c, a)
    return best

def fit_joint(om_grid, oq_grid, mode="a6"):
    zHD, *_ = H.load_sn()
    best = None
    for om in om_grid:
        for oq in oq_grid:
            cj = H.sn_chi2(Dc(zHD, om, oq, mode)) + H.bao_cmb_chi2(predict(om, oq, mode))[0]
            if best is None or cj < best[2]:
                best = (om, oq, cj)
    return best

def refine_joint(om0, oq0, mode="a6"):
    zHD, *_ = H.load_sn()
    def f(x):
        om, oq = x
        if om <= 0 or om >= 1.5:
            return 1e9
        return H.sn_chi2(Dc(zHD, om, oq, mode)) + H.bao_cmb_chi2(predict(om, oq, mode))[0]
    r = minimize(f, [om0, oq0], method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-3, maxiter=2000))
    return r.x[0], r.x[1], r.fun

# ---------------------------------------------------------------------------
def run_mode(mode, om_grid, oq_grid):
    print("="*74)
    print(f"MODE = {mode}")
    print("="*74)
    snS  = fit_sn_only(om_grid, oq_grid, mode)
    bcS  = fit_baocmb(om_grid, oq_grid, cmb=True, mode=mode)
    boS  = fit_baocmb(om_grid, oq_grid, cmb=False, mode=mode)
    jS0  = fit_joint(om_grid, oq_grid, mode)
    jS   = refine_joint(jS0[0], jS0[1], mode)
    _, aJ = H.bao_cmb_chi2(predict(jS[0], jS[1], mode))
    print(f"  SN-only : Om0={snS[0]:.3f} OQ0={snS[1]:+.3f} chi2={snS[2]:.1f} q0={q0_of(snS[0],snS[1]):+.2f}")
    print(f"  BAOonly : Om0={boS[0]:.3f} OQ0={boS[1]:+.3f} chi2={boS[2]:.1f} H0={H.H0_from_alpha(boS[3]):.1f}")
    print(f"  BAO+CMB : Om0={bcS[0]:.3f} OQ0={bcS[1]:+.3f} chi2={bcS[2]:.1f} H0={H.H0_from_alpha(bcS[3]):.1f} q0={q0_of(bcS[0],bcS[1]):+.2f}")
    print(f"  JOINT   : Om0={jS[0]:.3f} OQ0={jS[1]:+.3f} chi2={jS[2]:.1f} H0={H.H0_from_alpha(aJ):.1f} q0={q0_of(jS[0],jS[1]):+.2f}")
    return dict(sn=snS, baocmb=bcS, bao=boS, joint=jS, alpha=aJ)

if __name__ == "__main__":
    print("="*74)
    print("RASANEN peak-model / statistical backreaction (Buchert averaging)")
    print("="*74)
    print("VALIDATION:", validate())
    print()

    om_grid = np.linspace(0.05, 1.20, 116)
    oq_grid = np.linspace(-0.80, 0.30, 111)   # negative OQ -> q can be < 0

    # MODE late: backreaction confined to late times (high-z FRW recovery).
    R = run_mode("late", om_grid, oq_grid)

    # MODE a6: literal Buchert a^-6 closure (CMB term forces OQ0->0). For BAO+CMB
    # the a^-6 term blows up at z*, so restrict OQ0 grid near 0 there.
    print()
    R6 = run_mode("a6", om_grid, np.linspace(-0.05, 0.05, 41))

    # ---- peak-model published bound check ---------------------------------
    print("="*74)
    print("PEAK-MODEL PUBLISHED BOUND ([R08a]): |OQ0|<0.04, q0>0 (NO acceleration)")
    snP = fit_sn_only(om_grid, np.linspace(-0.04, 0.0, 9), "late")
    print(f"  SN within |OQ0|<=0.04: Om0={snP[0]:.3f} OQ0={snP[1]:+.3f} chi2={snP[2]:.1f} q0={q0_of(snP[0],snP[1]):+.2f}")

    # ---- LCDM reference ---------------------------------------------------
    def jl(Om):
        zHD,*_=H.load_sn(); return H.sn_chi2(H.lcdm_Dc(zHD,Om))+H.bao_cmb_chi2(H.lcdm_predict(Om))[0]
    rl = minimize_scalar(jl, bounds=(0.15,0.45), method="bounded")
    _, al = H.bao_cmb_chi2(H.lcdm_predict(rl.x))
    print("="*74)
    print(f"LCDM reference (harness): Om={rl.x:.3f} joint chi2={rl.fun:.1f} H0={H.H0_from_alpha(al):.1f}")
    import math
    lnN = math.log(1593)
    # BIC penalty: Rasanen joint floats k=2 cosmological params (Om0,OQ0) vs
    # LCDM's k=1 (Om); same (k-1)*ln(N) convention as model_ltbvoid.py:394 and
    # joint_w0wa.py (k=2 vs k=1 there too).
    dBIC_late = (R['joint'][2]-1402.2) + (2-1)*lnN
    dBIC_a6   = (R6['joint'][2]-1402.2) + (2-1)*lnN
    print(f"dBIC(joint late vs LCDM)  = {dBIC_late:+.1f}   (k=2 vs k=1, +{lnN:.2f} Occam penalty)")
    print(f"dBIC(joint a6   vs LCDM)  = {dBIC_a6:+.1f}")
    print(f"SN-only late OQ0={R['sn'][1]:+.3f} vs BAO+CMB late OQ0={R['baocmb'][1]:+.3f}  "
          f"(peak-model ceiling |OQ0|=0.04)")
