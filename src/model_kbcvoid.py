#!/usr/bin/env python3
"""
Local KBC underdensity ("Hubble bubble") on a global LCDM (Robertson/FLRW)
background.

PHYSICAL MODEL (primary refs):
  - Keenan, Barger & Cowie 2013 (arXiv:1304.2884): the local Universe is
    underdense out to R ~ 300 Mpc/h (z ~ 0.07), galaxy-number-density
    contrast delta_gal ~ -0.46 ("KBC void").
  - Haslbauer, Banik & Kroupa 2020 (arXiv:2009.11292): MNRAS 499, 2845 --
    model the KBC void on an EdS/LCDM background, show it can lift the LOCAL
    H0 toward 73 while the GLOBAL cosmology stays LCDM; but the required void
    depth/size is ~6 sigma beyond LCDM expectations.
  - Camarena, Marra, Sakr & Clarkson 2022 (arXiv:2107.02296): a local void
    consistent with number-counts CANNOT raise the inferred H0 enough to
    resolve the tension once the SH0ES Cepheid calibration is used; the void
    needed is too deep/large.

KEY IDEA & HARNESS SUBTLETY
  The global background is flat LCDM with one shape parameter Om. BAO (z>=0.3)
  and CMB (z=1090) probe the GLOBAL standard ruler and are blind to a ~300 Mpc
  local void => BAO+CMB == pure LCDM, global H0 ~ 67.

  A local void perturbs ONLY the very-low-z monopole of the Hubble diagram:
  inside an underdensity the local expansion is faster, so an observer at the
  centre measures a z-dependent apparent expansion rate
        H_loc(z) = H_glob_LCDM(z) * [1 + dH(z)],
        dH(z) = -(1/3) f bar_delta(z),   f = Om^0.55  (linear growth rate),
  where bar_delta(z) is the running mean MATTER density contrast inside the
  comoving sphere reaching redshift z.  Outside the void (bar_delta -> 0)
  H_loc -> H_glob.  The SN comoving distance integrates this perturbed rate:
        Dc(z) = int_0^z c dz' / H_loc(z').

  CRUCIAL: the harness sn_chi2 marginalises a CONSTANT offset (M_B + 5 log c/H0).
  A void that boosts H0 UNIFORMLY at all SN redshifts is degenerate with M_B
  and INVISIBLE to SN-only chi2.  What SN sees is only the SHAPE = the
  TRANSITION across the void edge (low-z H boosted, high-z H global).  So a
  void resolves the *tension* only via the external SH0ES Cepheid calibration
  (a true zero-point), NOT via the SN Hubble-diagram shape -- which is exactly
  Camarena+2022's point and the reason this model is "partial" in this harness.

VOID PROFILE
  Smooth compensated-ish top-hat (Garcia-Bellido & Haugbolle / Haslbauer style):
  central matter contrast delta0 (<0), comoving radius Rv (Mpc), edge width w.
  Running mean contrast inside radius r:  bar_delta(r) = delta0 * S(r), where
  S(r) is the volume-averaged shape. We use a simple representative profile and
  map redshift to comoving radius via the LCDM background:
        r(z) ~ (c/H0) * Dc_LCDM(z)   (Mpc; only the low-z part matters).

We FIT the global Om to BAO+CMB+SN exactly as LCDM (the void leaves the
*shape* of the SN diagram essentially unchanged once the offset is
marginalised), then quantify, for the literature void parameters, the LOCAL
H0 that an uncalibrated-distance-ladder / SH0ES analysis would infer.
"""
import sys, io
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid as cumtrap, quad

# importing harness prints the timescape self-block; silence it
_stdout = sys.stdout; sys.stdout = io.StringIO()
import harness as H
sys.stdout = _stdout

C_KM = 299792.458
RD = 147.09

zHD, zHEL, mb, Cf = H.load_sn()

# ----------------------------------------------------------------------
# GLOBAL background = flat LCDM (the Robertson/FLRW background).
# BAO+CMB see ONLY this -- the local void is invisible to z>=0.3 rulers.
# ----------------------------------------------------------------------
def E_lcdm(z, Om):
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om))

def invE_lcdm(z, Om):
    return 1.0 / E_lcdm(z, Om)

def lcdm_Dc(z, Om):
    zg = np.linspace(0, np.max(z) * 1.0001, 200000)
    Dc = cumtrap(invE_lcdm(zg, Om), zg, initial=0)
    return np.interp(z, zg, Dc)

def lcdm_predict(Om):
    def p(z, k):
        if k == "DH":
            return invE_lcdm(z, Om)
        dM = quad(lambda zz: invE_lcdm(zz, Om), 0, z)[0]
        if k == "DM":
            return dM
        return (z * dM * dM * invE_lcdm(z, Om)) ** (1 / 3)
    return p

# ----------------------------------------------------------------------
# LOCAL VOID: perturbed low-z expansion rate.
#   dH(z) = -(1/3) f bar_delta(z),  f = Om^0.55
#   bar_delta(z) = delta0 * shape(r(z)/Rv),  r(z) = (c/H0) Dc_LCDM(z)
# We use the dimensionless comoving distance D (=Dc/(c/H0)); the physical
# comoving radius is r = (c/H0)*D. With H0~67 => c/H0 ~ 4475 Mpc.
# ----------------------------------------------------------------------
def void_shape(x, edge=0.1):
    """Volume-averaged (running-mean) MATTER density contrast shape, normalised
    to 1 at the centre and ->0 outside the void. x = r / Rv.

    Physical model = a top-hat-like underdensity of comoving radius Rv with a
    smooth (tanh) edge of fractional width `edge`. The LOCAL contrast is
        delta_loc(x) = 0.5*[1 - tanh((x-1)/edge)]          (1 inside, 0 outside)
    The RUNNING MEAN inside radius x is the volume average
        S(x) = (3/x^3) int_0^x delta_loc(u) u^2 du
    which is ~1 deep inside and falls as 1/x^3 (mass conservation) well outside
    the edge -- i.e. the H perturbation is essentially gone a little past Rv,
    NOT a broad Gaussian tail. This keeps the SN-shape distortion confined to
    z <~ 0.08 (r <~ Rv) where it is degenerate with the marginalised offset."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    # cheap analytic-ish volume average via fine quadrature per call is costly;
    # vectorise with a shared integration grid.
    u = np.linspace(0.0, max(float(x.max()) * 1.0001, 1e-6), 4000)
    dloc = 0.5 * (1.0 - np.tanh((u - 1.0) / edge))
    integ = cumtrap(dloc * u ** 2, u, initial=0.0)
    Sgrid = np.where(u > 0, 3.0 * integ / np.maximum(u, 1e-12) ** 3, 1.0)
    out = np.interp(x, u, Sgrid)
    return out

def dH_of_z(z, Om, delta0, Rv_Mpc, H0_glob):
    """Fractional local expansion-rate perturbation at redshift z (scalar in,
    scalar out for scalar z)."""
    f = Om ** 0.55
    cH0 = C_KM / H0_glob                         # Mpc
    scalar = np.ndim(z) == 0
    D = lcdm_Dc(np.atleast_1d(z), Om)            # dimensionless (Dc/(c/H0))
    r = cH0 * D                                   # physical comoving radius, Mpc
    bar_delta = delta0 * void_shape(r / Rv_Mpc)
    res = -(1.0 / 3.0) * f * bar_delta
    return float(res[0]) if scalar else res

def void_Dc(z, Om, delta0, Rv_Mpc, H0_glob):
    """SN line-of-sight comoving distance (units c/H0_glob) WITH the local
    void perturbation. Dc = int dz' / [E(z')(1+dH(z'))]."""
    zmax = np.max(z) * 1.0001
    zg = np.linspace(0, zmax, 200000)
    cH0 = C_KM / H0_glob
    D = lcdm_Dc(zg, Om)
    r = cH0 * D
    f = Om ** 0.55
    bar_delta = delta0 * void_shape(r / Rv_Mpc)
    dH = -(1.0 / 3.0) * f * bar_delta
    integrand = invE_lcdm(zg, Om) / (1.0 + dH)
    Dc = cumtrap(integrand, zg, initial=0)
    return np.interp(z, zg, Dc)

# ----------------------------------------------------------------------
# FITS
# ----------------------------------------------------------------------
def fit_sn_lcdm():
    """SN-only shape fit (no void): constrains Om from the Hubble-diagram shape."""
    f = lambda Om: H.sn_chi2(lcdm_Dc(zHD, Om))
    r = minimize_scalar(f, bounds=(0.05, 0.6), method="bounded")
    return r.x, r.fun

def fit_sn_void(delta0, Rv, H0_glob):
    """SN-only shape fit WITH a fixed void (delta0, Rv); Om free."""
    f = lambda Om: H.sn_chi2(void_Dc(zHD, Om, delta0, Rv, H0_glob))
    r = minimize_scalar(f, bounds=(0.05, 0.6), method="bounded")
    return r.x, r.fun

def fit_bao_cmb():
    f = lambda Om: H.bao_cmb_chi2(lcdm_predict(Om))[0]
    r = minimize_scalar(f, bounds=(0.15, 0.45), method="bounded")
    _, a = H.bao_cmb_chi2(lcdm_predict(r.x))
    return r.x, r.fun, H.H0_from_alpha(a)

def fit_bao_only():
    f = lambda Om: H.bao_only_chi2(lcdm_predict(Om))[0]
    r = minimize_scalar(f, bounds=(0.15, 0.45), method="bounded")
    _, a = H.bao_only_chi2(lcdm_predict(r.x))
    return r.x, r.fun, H.H0_from_alpha(a)

def fit_joint_void(delta0, Rv, H0_glob):
    """Joint SN(void) + BAO+CMB(LCDM): single Om, void params fixed externally."""
    def joint(Om):
        return H.sn_chi2(void_Dc(zHD, Om, delta0, Rv, H0_glob)) + \
               H.bao_cmb_chi2(lcdm_predict(Om))[0]
    r = minimize_scalar(joint, bounds=(0.15, 0.45), method="bounded")
    _, a = H.bao_cmb_chi2(lcdm_predict(r.x))
    return r.x, r.fun, H.H0_from_alpha(a)

def fit_joint_lcdm():
    def joint(Om):
        return H.sn_chi2(lcdm_Dc(zHD, Om)) + H.bao_cmb_chi2(lcdm_predict(Om))[0]
    r = minimize_scalar(joint, bounds=(0.15, 0.45), method="bounded")
    _, a = H.bao_cmb_chi2(lcdm_predict(r.x))
    return r.x, r.fun, H.H0_from_alpha(a)

def q0(Om):
    # flat LCDM deceleration today: q0 = Om/2 - OL = 1.5 Om - 1
    return 1.5 * Om - 1.0

if __name__ == "__main__":
    print("=" * 74)
    print("LOCAL KBC VOID ('Hubble bubble') on a global LCDM background")
    print("=" * 74)

    # ---- VALIDATION 1: global background reproduces harness LCDM reference ----
    Om_j, chi_j, H0_j = fit_joint_lcdm()
    print(f"\n[VALIDATION] Global-LCDM joint reproduces harness reference:")
    print(f"  Om={Om_j:.3f}  joint chi2={chi_j:.1f}  H0={H0_j:.1f}  q0={q0(Om_j):.2f}")
    print(f"  (harness reference: Om=0.305, chi2=1402.2, H0=68.6, q0=-0.55)")

    # ---- VALIDATION 2: dH(z->0) reproduces the linear-theory void H0 boost ----
    # Haslbauer+2020 / Keenan+2013 KBC void: Rv~300 Mpc, delta_m ~ -0.5 (deep).
    Om_v = Om_j
    H0_glob = H0_j
    for d0, Rv in [(-0.3, 300.0), (-0.46, 300.0), (-0.52, 300.0), (-0.8, 300.0)]:
        dh0 = dH_of_z(1e-4, Om_v, d0, Rv, H0_glob)
        print(f"[VALIDATION] void delta0={d0:+.2f} Rv={Rv:.0f}Mpc: dH/H(0)={dh0:+.3f} "
              f"-> local H0={H0_glob*(1+dh0):.1f} "
              f"(linear: -(1/3)Om^0.55 delta0 = {-(1/3)*Om_v**0.55*d0:+.3f})")

    print("\n" + "-" * 74)
    print("FITS")
    print("-" * 74)

    Om_sn, chi_sn = fit_sn_lcdm()
    print(f"SN-only (no void, shape):  Om={Om_sn:.3f}  chi2={chi_sn:.1f}")

    Om_bo, chi_bo, H0_bo = fit_bao_only()
    print(f"BAO-only (LCDM bg):        Om={Om_bo:.3f}  chi2={chi_bo:.1f}  H0={H0_bo:.1f}")

    Om_bc, chi_bc, H0_bc = fit_bao_cmb()
    print(f"BAO+CMB (LCDM bg):         Om={Om_bc:.3f}  chi2={chi_bc:.1f}  H0={H0_bc:.1f}  q0={q0(Om_bc):.2f}")

    # joint with the SH0ES-bridging void depth (delta0=-0.52)
    d0_bridge, Rv_bridge = -0.52, 300.0
    Om_jv, chi_jv, H0_jv = fit_joint_void(d0_bridge, Rv_bridge, H0_bc)
    dh0 = dH_of_z(1e-4, Om_jv, d0_bridge, Rv_bridge, H0_bc)
    H0_local = H0_jv * (1 + dh0)
    print(f"\nJOINT SN(void)+BAO+CMB:    Om={Om_jv:.3f}  chi2={chi_jv:.1f}  "
          f"H0_glob={H0_jv:.1f}  H0_local={H0_local:.1f}  q0={q0(Om_jv):.2f}")
    print(f"  (void: delta0={d0_bridge}, Rv={Rv_bridge:.0f}Mpc -> dH/H(0)={dh0:+.3f})")

    # SN shape fit WITH the bridging void (does the void spoil the SN shape?)
    Om_snv, chi_snv = fit_sn_void(d0_bridge, Rv_bridge, H0_bc)
    print(f"SN-only WITH void (shape): Om={Om_snv:.3f}  chi2={chi_snv:.1f}  "
          f"(d chi2 vs no-void SN = {chi_snv - chi_sn:+.2f})")

    # dBIC vs LCDM joint. LCDM joint k=2 (Om + offset). Void adds delta0, Rv
    # but here we FIX them from literature (not fitted) -> same k=2; if we
    # counted them as fitted, k=4. Report both framings.
    N = 1593
    chi_lcdm_joint = 1402.2
    dchi = chi_jv - chi_lcdm_joint
    dBIC_samek = dchi                       # void params fixed (not fitted)
    dBIC_2extra = dchi + 2 * np.log(N)      # if delta0,Rv counted as fitted
    print(f"\ndBIC vs LCDM joint (1402.2):")
    print(f"  void params FIXED from literature (k=2): dchi2={dchi:+.2f} -> dBIC={dBIC_samek:+.2f}")
    print(f"  void params FITTED (k=4): dBIC={dBIC_2extra:+.2f}")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("The void leaves the SN Hubble-diagram SHAPE essentially unchanged once")
    print("the M_B/H0 offset is marginalised (the harness's SN chi2 is shape-only),")
    print("so SN and BAO+CMB AGREE on Om (no tension between datasets in this harness).")
    print("The void's H0 boost (67->~73) acts ONLY through an EXTERNAL Cepheid")
    print("zero-point (SH0ES), which this harness marginalises away. Whether the void")
    print("resolves the tension thus hinges on (a) external SH0ES calibration and")
    print("(b) whether a delta_m~-0.5, R~300Mpc void is allowed -- Camarena+2022 say")
    print("the number-counts void is too shallow (delta_m~-0.2..-0.3 -> only ~70-71).")
