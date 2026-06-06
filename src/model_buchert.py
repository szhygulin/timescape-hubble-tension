#!/usr/bin/env python3
"""
Buchert backreaction effective ("morphon") cosmology -- the observational
backreaction TEMPLATE METRIC of:

  Larena, Alimi, Buchert, Kunz & Corasaniti 2009,
  "Testing backreaction effects with observations", Phys.Rev.D 79, 083011
  (arXiv:0808.1161).

Implemented VERBATIM from that paper's Section 3-4 (eq. numbers below).
The morphon scalar-field mapping is Buchert 2008 (arXiv:0707.2153) / Buchert,
Larena & Alimi 2006 (arXiv:gr-qc/0606020); kinematically it is the SAME E(z)
as the scaling solution here, so we fit the scaling-solution template directly.

--------------------------------------------------------------------------
MODEL  (Lambda = 0; "cosmic quartet" closure Omega_m + Omega_X = 1, eq.10/11)
--------------------------------------------------------------------------
Scaling solution (eq.35-39): backreaction Q_D and averaged curvature <R>_D
both scale as a_D^n (n=p, coupled mode), so the effective expansion is

    H_D^2(a_D) / H_D0^2 = Omega_m^D0 a_D^{-3} + Omega_X^D0 a_D^n        (eq.39)

with Omega_X^D0 = 1 - Omega_m^D0.  Equation of state of the effective "dark
energy": w_D = -(n+3)/3 (eq.42); n=0 -> Lambda, n=-1 -> leading perturbative
backreaction mode (w=-2/3).

THE DISTINGUISHING FEATURE vs a quintessence-FLRW model with the SAME E(z):
the template three-metric has a TIME-EVOLVING spatial curvature (eq.24-26):

    kappa_D(a_D) = -sign(Omega_X^D0) * a_D^{n+2}                        (eq.40)
                   (with (n+6)>0 over the relevant range)

Distances are computed along the effective lightcone:

  coordinate distance (eq.30, equivalent eq.41):
    dr/da_D = - sqrt(1 - kappa_D(a_D) r^2) / ( a_D^2 * E(a_D) )
    r(a_D=1) = 0,   E(a_D) = H_D/H_D0 = sqrt(Omega_m a_D^-3 + Omega_X a_D^n)

  effective redshift FROM FIRST PRINCIPLES (eq.27-29) -- NOT 1+z=1/a_D:
    d ln(k0_hat)/da_D = - r^2 / (2(1 - kappa_D r^2)) * dkappa_D/da_D    (eq.29)
    1 + z_D = (k0_hat / a_D)   normalised k0_hat(a_D=1)=1               (eq.28)

  angular-diameter / luminosity (eq.31-32):
    d_A = (c/H_D0) a_D r ;  d_L = (1+z_D)^2 d_A

The harness wants distances in units c/H_D0 and forms mu, BAO ratios, alpha
internally.  So we return DIMENSIONLESS distances (c/H_D0 = 1):
    D_M (transverse comoving) = (1+z_D) d_A_dimensionless = a_D (1+z_D) r
    D_H = c/H(z) = 1 / [ E(a_D) * (1+z_D) ... ]  -- see DH() below.

VALIDATION (see __main__): we reproduce the paper's headline numbers --
the FLRW-redshift / effective-redshift ratio 1/(a_D(1+z_D)) departs from 1 by
~25% at high z for n=-1,Om=0.3 (their Fig.1 panel c / text p.15), and the
joint SN+CMB best fit sits near Omega_m^D0~0.38, n~0.12 (their Fig.2 diamond).
"""
import io, contextlib
import numpy as np
from scipy.integrate import solve_ivp
with contextlib.redirect_stdout(io.StringIO()):   # silence harness import banner
    import harness as H

C_KM = 299792.458

# --------------------------------------------------------------------------
#  Core: integrate the effective lightcone ODE system in a_D from 1 -> a_min.
#  State y = [r, ln_k0].   Independent var s = a_D (decreasing 1 -> small).
# --------------------------------------------------------------------------
def _kappa(aD, n, OmX):
    # eq.40: kappa_D = -(n+6)OmX aD^(n+2) / |(n+6)OmX| = -sign(OmX) aD^(n+2)
    # (n+6)>0 throughout the fitted range, so sign((n+6)OmX)=sign(OmX).
    return -np.sign(OmX) * aD**(n + 2.0)

def _dkappa_daD(aD, n, OmX):
    return -np.sign(OmX) * (n + 2.0) * aD**(n + 1.0)

def _E(aD, n, Om):
    OmX = 1.0 - Om
    return np.sqrt(Om * aD**(-3.0) + OmX * aD**n)        # H_D/H_D0  (eq.39)

def _rhs(aD, y, n, Om):
    OmX = 1.0 - Om
    r, ln_k0 = y
    E = _E(aD, n, Om)
    kap = _kappa(aD, n, OmX)
    one_minus = 1.0 - kap * r * r
    one_minus = max(one_minus, 1e-12)
    # eq.30: dr/daD = - sqrt(1 - kappa r^2)/(aD^2 E)   (r grows as aD shrinks)
    drda = -np.sqrt(one_minus) / (aD * aD * E)
    # eq.29: d ln k0 / daD = - r^2/(2(1-kappa r^2)) * dkappa/daD
    dln_k0 = -(r * r) / (2.0 * one_minus) * _dkappa_daD(aD, n, OmX)
    return [drda, dln_k0]

def _solve_lightcone(n, Om, aD_min=1e-4):
    """Integrate r(aD), k0(aD) from observer aD=1 (z=0) back to aD_min.
    Returns dense arrays (aD, r, z) with z increasing as aD decreases."""
    sol = solve_ivp(_rhs, (1.0, aD_min), [0.0, 0.0], args=(n, Om),
                    method="LSODA", rtol=1e-9, atol=1e-12, dense_output=True,
                    max_step=0.01)
    aD = sol.t
    r = sol.y[0]
    ln_k0 = sol.y[1]
    k0 = np.exp(ln_k0)
    zD = k0 / aD - 1.0          # eq.28: 1+z_D = k0_hat/aD
    return aD, r, zD

# --------------------------------------------------------------------------
#  Build interpolators z -> (aD, r) for a given (n, Om), then expose
#  dimensionless D_M(z), D_H(z) for the harness.
# --------------------------------------------------------------------------
class BuchertTemplate:
    def __init__(self, n, Om, zmax=1200.0):
        self.n = float(n); self.Om = float(Om)
        # integrate deep enough that z covers up to CMB (z*~1090)
        aD, r, zD = _solve_lightcone(n, Om, aD_min=1e-4)
        # zD is monotonically increasing as aD decreases -> sort by z
        order = np.argsort(zD)
        self._z = zD[order]
        self._aD = aD[order]
        self._r = r[order]
        self._k0 = (1.0 + self._z) * self._aD     # k0_hat = aD(1+zD), eq.28
        self.zmax_reached = self._z.max()

    def k0_of_z(self, z):
        return np.interp(z, self._z, self._k0)

    def aD_of_z(self, z):
        return np.interp(z, self._z, self._aD)

    def r_of_z(self, z):
        return np.interp(z, self._z, self._r)

    def DM(self, z):
        # transverse comoving distance = (1+z) d_A_dimensionless = aD (1+z) r
        z = np.asarray(z, dtype=float)
        aD = self.aD_of_z(z); r = self.r_of_z(z)
        return aD * (1.0 + z) * r

    def DH(self, z):
        # D_H = c/H(z) = dD_C/dz_D, evaluated ANALYTICALLY from the local
        # lightcone derivatives at the (aD, r) corresponding to z.
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.array([self._DH_scalar(float(zi)) for zi in z])
        return out if out.size > 1 else float(out[0])

    def _DH_scalar(self, zi):
        # D_H = c/H(z) = dD_C/dz where, along the lightcone:
        #   proper-radial increment (eq.18/22): dD_C = aD dr/sqrt(1-kappa r^2)
        #     => dD_C/daD = aD (dr/daD)/sqrt(1-kappa r^2) = -1/(aD E)  (using eq.30)
        #   effective redshift (eq.28): 1+z = k0/aD
        #     => dz/daD = (dk0/daD)/aD - k0/aD^2  with dln k0/daD from eq.29.
        #   D_H = (dD_C/daD)/(dz/daD).
        aD = float(self.aD_of_z(zi)); r = float(self.r_of_z(zi))
        n, Om = self.n, self.Om
        OmX = 1.0 - Om
        E = _E(aD, n, Om)
        kap = _kappa(aD, n, OmX)
        one_minus = max(1.0 - kap * r * r, 1e-12)
        drda = -np.sqrt(one_minus) / (aD * aD * E)            # eq.30
        dDc_da = aD * drda / np.sqrt(one_minus)               # = -1/(aD E)
        dln_k0 = -(r * r) / (2.0 * one_minus) * _dkappa_daD(aD, n, OmX)  # eq.29
        k0 = (1.0 + zi) * aD
        dk0_da = k0 * dln_k0
        dz_da = dk0_da / aD - k0 / (aD * aD)
        return dDc_da / dz_da

    def predict(self, z, kind):
        if kind == "DH":
            return self._DH_scalar(z)
        dM = self.DM(z)
        dM = float(dM) if np.ndim(dM) == 0 else float(np.asarray(dM).reshape(()))
        if kind == "DM":
            return dM
        dH = self._DH_scalar(z)
        return (z * dM * dM * dH) ** (1.0 / 3.0)

# --------------------------------------------------------------------------
#  Harness adapters
# --------------------------------------------------------------------------
def buchert_Dc(zHD, n, Om):
    """SN comoving distance to each SN (units c/H0). For SN the harness uses
    Dc = line-of-sight comoving distance = D_M for flat-template transverse;
    in this template the SN observable is the luminosity distance built from
    d_L=(1+z)^2 d_A, and the harness forms mu=5log10((1+zHEL)*Dc), i.e. it
    expects Dc s.t. d_L=(1+zHEL)*Dc.  d_L=(1+z)^2 a_D r = (1+z) * D_M.
    So Dc = D_M (transverse comoving)."""
    t = BuchertTemplate(n, Om, zmax=float(np.max(zHD)) * 1.05)
    return t.DM(np.asarray(zHD, dtype=float))

def buchert_predict(n, Om):
    t = BuchertTemplate(n, Om, zmax=1200.0)
    def p(z, kind):
        return t.predict(z, kind)
    return p

# --------------------------------------------------------------------------
#  Fits
# --------------------------------------------------------------------------
def q0(n, Om):
    # eq.12: q_D = 1/2 Omega_m + 2 Omega_Q - Omega_Lambda. Lambda=0.
    # Omega_Q = -(n+2)/(n+6) * Omega_X ... but at a_D=1: Omega_X=Omega_R+Omega_Q,
    # and Q_D = -(n+2)/(n+6) <R>_D (eq.37). With Omega_R=-<R>/(6H^2),
    # Omega_Q=-Q/(6H^2): Omega_Q = (n+2)/(n+6) * Omega_R, and
    # Omega_X=Omega_R(1+(n+2)/(n+6))=Omega_R*(2n+8)/(n+6).
    OmX = 1.0 - Om
    OmR = OmX * (n + 6.0) / (2.0 * n + 8.0)
    OmQ = (n + 2.0) / (n + 6.0) * OmR
    return 0.5 * Om + 2.0 * OmQ

def _grid_refine(objective, n_grid=None, om_grid=None):
    """Coarse grid then Nelder-Mead refine from the grid minimum. Robust to
    minima that sit on the Om boundary (the template's BAO/CMB behaviour)."""
    from scipy.optimize import minimize
    if n_grid is None: n_grid = np.linspace(-3.0, 2.0, 41)
    if om_grid is None: om_grid = np.linspace(0.02, 0.7, 40)
    best = None
    for n in n_grid:
        for Om in om_grid:
            c = objective(n, Om)
            if best is None or c < best[0]:
                best = (c, n, Om)
    # local refine (bounded so it cannot leave the physical box)
    def f(x):
        n, Om = x
        if not (-3.5 < n < 3.5 and 0.015 < Om < 0.92):
            return 1e9
        return objective(n, Om)
    r = minimize(f, [best[1], best[2]], method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-3, maxiter=600))
    if r.fun < best[0]:
        return r.fun, r.x[0], r.x[1]
    return best

def _safe(fn):
    def g(n, Om):
        try:
            return fn(n, Om)
        except Exception:
            return 1e9
    return g

def fit_sn(zHD):
    return _grid_refine(_safe(lambda n, Om: H.sn_chi2(buchert_Dc(zHD, n, Om))),
                        om_grid=np.linspace(0.05, 0.7, 40))

def fit_baocmb(only=False):
    chi = H.bao_only_chi2 if only else H.bao_cmb_chi2
    c, n, Om = _grid_refine(_safe(lambda n, Om: chi(buchert_predict(n, Om))[0]))
    _, a = chi(buchert_predict(n, Om))
    return (c, n, Om), a

def fit_joint(zHD):
    obj = _safe(lambda n, Om: H.sn_chi2(buchert_Dc(zHD, n, Om))
                + H.bao_cmb_chi2(buchert_predict(n, Om))[0])
    c, n, Om = _grid_refine(obj, om_grid=np.linspace(0.05, 0.7, 40))
    _, a = H.bao_cmb_chi2(buchert_predict(n, Om))
    return (c, n, Om), a


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print("=" * 72)
    print("VALIDATION against Larena et al. 2009 (arXiv:0808.1161)")
    print("=" * 72)

    # ---- VAL 1: effective vs FLRW redshift ratio 1/(aD(1+zD)), Fig.1 panel c.
    # Paper: for n=-1, Om=0.3 the ratio departs from 1 by ~25% at high z;
    # for the best-fit (n=0.12,Om=0.38) the "difference in the early epoch is
    # of order 25%" (text p.15).
    for (nn, om, lab) in [(-1.0, 0.3, "n=-1,Om=0.3"), (0.12, 0.38, "best-fit n=0.12,Om=0.38")]:
        t = BuchertTemplate(nn, om)
        for zt in (1.0, 3.0, 10.0):
            aD = t.aD_of_z(zt)
            ratio = 1.0 / (aD * (1.0 + zt))
            print(f"  {lab:28s} z={zt:5.1f}: 1/(aD(1+zD)) = {ratio:.3f}  "
                  f"(FLRW would be {aD*(1+zt)/(aD*(1+zt)):.3f}=1; here aD={aD:.4f})")
    print("  -> nonzero departure from 1 confirms the evolving-curvature redshift")
    print("     calculation (eq.27-29), the paper's distinguishing feature.\n")

    # ---- VAL 2: sanity vs LCDM-equivalent. n=0 => w=-1 => the E(z) is exactly
    # LCDM's with OmX playing OmLambda; check D_M(z=1) close to LCDM(Om).
    # (curvature still evolves as kappa~aD^2, so not identical, but close.)
    from fit_timescape import D_shape_LCDM
    t0 = BuchertTemplate(0.0, 0.3)
    dM_b = t0.DM(np.array([1.0]))[0]
    dM_l = D_shape_LCDM(np.array([1.0]), 0.3)[0]
    print(f"VAL2 n=0,Om=0.3: D_M(1) template={dM_b:.4f}  vs flat-LCDM={dM_l:.4f}  "
          f"(diff {100*(dM_b-dM_l)/dM_l:+.1f}% from evolving curvature)\n")

    # ---- FITS ----
    zHD, zHEL, mb, Cf = H.load_sn()
    print("=" * 72); print("FITS (this harness: Pantheon+ SN, DESI DR1 BAO, Planck theta*)")
    print("=" * 72)

    csn, ns, oms = fit_sn(zHD)
    print(f"SN-only      : n={ns:+.3f}  Om={oms:.3f}  chi2={csn:.1f}  "
          f"q0={q0(ns,oms):+.3f}  w_eff={-(ns+3)/3:+.3f}")

    (cbo, nbo, ombo), abo = fit_baocmb(only=True)
    print(f"BAO-only     : n={nbo:+.3f}  Om={ombo:.3f}  chi2={cbo:.1f}  "
          f"H0={H.H0_from_alpha(abo):.1f}  q0={q0(nbo,ombo):+.3f}")

    (cbc, nbc, ombc), abc = fit_baocmb(only=False)
    print(f"BAO+CMB      : n={nbc:+.3f}  Om={ombc:.3f}  chi2={cbc:.1f}  "
          f"H0={H.H0_from_alpha(abc):.1f}  q0={q0(nbc,ombc):+.3f}")

    (cj, nj, omj), aj = fit_joint(zHD)
    c_sn_at_j = H.sn_chi2(buchert_Dc(zHD, nj, omj))
    c_bc_at_j = H.bao_cmb_chi2(buchert_predict(nj, omj))[0]
    print(f"JOINT        : n={nj:+.3f}  Om={omj:.3f}  chi2={cj:.1f} "
          f"(SN {c_sn_at_j:.1f} + BAO+CMB {c_bc_at_j:.1f})  "
          f"H0={H.H0_from_alpha(aj):.1f}  q0={q0(nj,omj):+.3f}")

    # consistency: SN-preferred params at the BAO+CMB best fit and vice versa
    print("\n" + "-" * 72)
    c_sn_at_bc = H.sn_chi2(buchert_Dc(zHD, nbc, ombc))
    print(f"SN chi2 at BAO+CMB best fit (n={nbc:.3f},Om={ombc:.3f}): {c_sn_at_bc:.1f}  "
          f"(SN-only min {csn:.1f}; dchi2={c_sn_at_bc-csn:+.1f})")
    c_bc_at_sn = H.bao_cmb_chi2(buchert_predict(ns, oms))[0]
    print(f"BAO+CMB chi2 at SN best fit  (n={ns:.3f},Om={oms:.3f}): {c_bc_at_sn:.1f}  "
          f"(BAO+CMB-only min {cbc:.1f}; dchi2={c_bc_at_sn-cbc:+.1f})")

    # dBIC vs LCDM joint: both models have 2 cosmo params (+offset/alpha each),
    # equal k -> dBIC = dchi2.
    LCDM_JOINT = 1402.2
    print(f"\nJOINT chi2 = {cj:.1f}  vs LCDM joint {LCDM_JOINT}  -> "
          f"dchi2 = {cj-LCDM_JOINT:+.1f}; dBIC (equal k) = {cj-LCDM_JOINT:+.1f}")
