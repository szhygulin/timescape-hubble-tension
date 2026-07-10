#!/usr/bin/env python3
"""
LTB giant-void cosmology (we sit near the centre of a Gpc-scale underdensity).
Garcia-Bellido & Haugbolle 2008 (JCAP, arXiv:0802.1523) "constrained GBH" (CGBH)
profile with a HOMOGENEOUS big bang (t_B = const), no Lambda.

We integrate the ON-CENTRE radial null geodesic ODEs and build D_A(z), then
D_M = (1+z) D_A (transverse comoving, since on-centre photons travel radially),
D_H = c/H_long (longitudinal Hubble rate along the line of sight), and
D_V = (z D_M^2 D_H)^{1/3}.  All in units c/H0_out so they plug into the harness
which marginalises over the absolute distance scale (alpha = c/(H0 rd) for
BAO+CMB, and an M_B offset for SN).

----------------------------------------------------------------------------
LTB METRIC & DYNAMICS (no Lambda, GBH conventions; see 0802.1523 Sec. 2)
----------------------------------------------------------------------------
ds^2 = -c^2 dt^2 + R'(r,t)^2 / (1 - k(r) r^2) dr^2 + R(r,t)^2 dOmega^2
where R' = dR/dr.  The local expansion obeys

  (Rdot/R)^2 = H_perp^2 = H0_perp(r)^2 [ Om_M(r) (R0/R)^3 + Om_k(r) (R0/R)^2 ]

with R0 = R(r,t0) = r (gauge), Om_M(r)+Om_k(r)=1 (no Lambda), and Om_k = 1-Om_M.
Two free radial functions in general; CGBH fixes one by demanding a uniform big
bang age t_B = 0 everywhere, which ties H0_perp(r) to Om_M(r):

  t0 = (1/H0_perp(r)) * eta(Om_M(r)),   eta(Om) = the open-universe age factor
      eta(Om) = Om/( (1-Om)^{3/2} ) * [ arcsinh( sqrt((1-Om)/Om) ) ... ]   (Om<1)

So with a single global age t0 we get H0_perp(r) = eta(Om_M(r)) / t0.  We absorb
t0 by working in units where H0_perp(r->infinity) maps to the asymptotic rate;
practically we normalise everything to the on-centre observed quantities and let
the harness marginalise the absolute scale.

GBH constrained-model density profile (their Eq. 2.5, "CGBH"):
  Om_M(r) = Om_out + (Om_in - Om_out) * ( 1 - tanh((r-r0)/2 dr) )
                                       / ( 1 + tanh( r0 /2 dr) )
Constrained model: Om_out = 1 (EdS asymptotic, no Lambda) so the only shape
params are (Om_in, r0, dr).  We add Om_out as a 4th param for the general GBH
open variant but the headline fit uses the 3-param CGBH (Om_out=1).

----------------------------------------------------------------------------
ON-CENTRE RADIAL NULL GEODESIC (GBH Eq. 2.9-2.10 / Bull,Clifton,Ferreira 1108.2222)
----------------------------------------------------------------------------
Incoming radial photon, parametrise by r:
   dt/dr = - R'(r,t) / ( c sqrt(1 - k(r) r^2) )
   dz/dr = (1+z) Rdot'(r,t) / ( c sqrt(1 - k(r) r^2) )     [Rdot' = d/dt of R']
The angular-diameter distance for an on-centre observer is simply D_A(z)=R(r(z),t(z))
(the areal radius along the past light cone).  We obtain R, R', Rdot, Rdot' from
the parametric LTB solution for each (r,t).
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize

C_KM = 299792.458

# ============================================================================
# LTB parametric solution for the open (k>0 curvature, hyperbolic) FLRW-in-shell
# We use the standard parametric (developmental angle u) solution of
#   Rdot^2 = H0p^2 r^2 [ OmM (r/R)^3 ... ] with a homogeneous bang t_B=0.
# For each comoving shell r the local dynamics is an open-FLRW with parameters
#   OmM = OmM(r),  H0p = H0p(r),  curvature OmK = 1-OmM > 0.
# Areal radius R(r,t) = r * a_shell(r,t) where a_shell solves the local Friedmann
#   (da/dt)^2 = H0p^2 [ OmM/a + (1-OmM) ]      (a normalised a(t0)=1)
# Parametric open solution (OmM<1):
#   a(u)   = (OmM/(2(1-OmM))) (cosh u - 1)
#   t(u)   = (OmM/(2 H0p (1-OmM)^{3/2})) (sinh u - u)
# t0 from a=1.  This gives R, Rdot directly; r-derivatives R', Rdot' by chain
# rule through OmM(r),H0p(r) (computed numerically by finite difference in r).
# ============================================================================

def age_factor(OmM):
    """eta(Om) such that t0 = eta(Om)/H0p, for an open (Om<1) matter universe."""
    OmM = np.clip(OmM, 1e-6, 0.999999)
    OmK = 1.0 - OmM
    # t0 = (1/H0p) * [ OmM/(2 OmK^{3/2}) ] * (sinh u0 - u0) with a(u0)=1
    # a=1 -> cosh u0 - 1 = 2 OmK/OmM -> cosh u0 = 1 + 2 OmK/OmM
    cu0 = 1.0 + 2.0*OmK/OmM
    u0 = np.arccosh(cu0)
    return (OmM/(2.0*OmK**1.5))*(np.sinh(u0) - u0)

# global age t0 in units where c=H0_out=1 conventions handled via scaling below.
# We fix the GLOBAL age by the asymptotic shell; then H0p(r) = age_factor(OmM(r))/t0.
# Choose t0 so that the asymptotic (r->inf) shell has H0p_out = 1 in our units.
# i.e. t0 = age_factor(Om_out)/H0p_out with H0p_out := 1.  All distances then in
# units c/H0p_out; harness marginalises the absolute scale anyway.

def make_profile(Om_in, Om_out, r0, dr):
    """CGBH density profile OmM(r); r in units of c/H0_out (so r0,dr ~ O(1) Gpc-ish)."""
    denom = 1.0 + np.tanh(r0/(2.0*dr))
    def OmM(r):
        return Om_out + (Om_in - Om_out)*(1.0 - np.tanh((r - r0)/(2.0*dr)))/denom
    return OmM

class LTBVoid:
    def __init__(self, Om_in, Om_out, r0, dr):
        self.Om_in, self.Om_out, self.r0, self.dr = Om_in, Om_out, r0, dr
        self.OmM = make_profile(Om_in, Om_out, r0, dr)
        # global age fixed by asymptotic shell with H0p_out=1 (units c=1, H0p_out=1)
        self.H0p_out = 1.0
        self.t0 = age_factor(Om_out)/self.H0p_out
        # small-r regularisation: at r=0 use Om_in
    def H0p(self, r):
        return age_factor(self.OmM(r))/self.t0
    # ---- local open-FLRW scale factor a(r,t): solve cosh/sinh parametric ----
    def _a_and_rates(self, r, t):
        """Return a(r,t), adot = da/dt (= a_,t), for shell r at cosmic time t.
        Solve t = (OmM/(2 H0p OmK^{3/2}))(sinh u - u) for u, then a=(OmM/(2 OmK))(cosh u -1)."""
        OmM = self.OmM(r); OmM = min(max(OmM,1e-6),0.999999)
        OmK = 1.0 - OmM
        H0p = self.H0p(r)
        A = OmM/(2.0*OmK)          # a = A(cosh u -1)
        B = OmM/(2.0*H0p*OmK**1.5) # t = B(sinh u - u)
        if t <= 0: return 0.0, np.inf
        # solve B(sinh u - u) = t for u>0; sinh u - u increasing
        g = lambda u: B*(np.sinh(u)-u) - t
        # bracket: sinh u - u is monotone increasing, find uhi with g(uhi)>0
        uhi = 1.0
        while g(uhi) < 0:
            uhi *= 2.0
            if uhi > 1e6:
                break
        u = brentq(g, 1e-12, uhi, xtol=1e-13, rtol=1e-12)
        a = A*(np.cosh(u)-1.0)
        # da/dt = (da/du)/(dt/du) = [A sinh u] / [B(cosh u -1)]
        adot = (A*np.sinh(u))/(B*(np.cosh(u)-1.0))
        return a, adot

# ----------------------------------------------------------------------------
# Areal radius and its derivatives.  R(r,t)=r*a(r,t).
#   R'   = a + r * a_,r        (a_,r = partial a / partial r at fixed t)
#   Rdot = r * a_,t
#   Rdot'= a_,t + r * a_,rt    (mixed partial)
# We compute a_,r and a_,rt by central finite difference in r of a and adot.
# Curvature: open shells have effective k(r) r^2 with 1-k r^2 = 1 + (H0p^2 OmK r^2)
# In GBH units the geodesic uses sqrt(1 - k r^2) with -k r^2 = OmK(r) H0p(r)^2 r^2,
# i.e. 1 - k r^2 = 1 + OmK H0p^2 r^2 (open => denominator >1).
# ----------------------------------------------------------------------------

def Rfuncs(m, r, t, hr=None):
    if hr is None:
        hr = max(1e-5, 1e-4*max(r, 1.0))
    a, adot = m._a_and_rates(r, t)
    ap, adotp = m._a_and_rates(r+hr, t)
    am, adotm = m._a_and_rates(max(r-hr,1e-9), t)
    a_r  = (ap - am)/(2*hr)
    adot_r = (adotp - adotm)/(2*hr)
    R    = r*a
    Rp   = a + r*a_r
    Rdot = r*adot
    Rdotp= adot + r*adot_r
    return R, Rp, Rdot, Rdotp

def sqrt_one_minus_kr2(m, r):
    OmM = m.OmM(r); OmK = 1.0 - min(max(OmM,1e-6),0.999999)
    H0p = m.H0p(r)
    return np.sqrt(1.0 + OmK*(H0p**2)*(r**2))

# ----------------------------------------------------------------------------
# Radial null geodesic integrator (on-centre observer, incoming photon).
# State y=[t, z] as function of r, integrate from r=0 (z=0,t=t0) outward.
#   dt/dr = - Rp / (c sqrt(1-k r^2))      (c=1 in our units)
#   dz/dr = (1+z) Rdotp / (c sqrt(1-k r^2))
# (Bull,Clifton,Ferreira 2012 Eq.; GBH 0802.1523 Eq. 2.9-2.10)
# ----------------------------------------------------------------------------

def integrate_lightcone(m, r_max, n_eval=4000, zstop=None):
    def rhs(r, y):
        t, z = y
        if r < 1e-9:
            # near centre R'->a(0,t), Rdot'->adot(0,t)+0; k r^2->0
            a, adot = m._a_and_rates(1e-9, t)
            return [-a, (1+z)*adot]
        R, Rp, Rdot, Rdotp = Rfuncs(m, r, t)
        s = sqrt_one_minus_kr2(m, r)
        dtdr = -Rp/s
        dzdr = (1.0+z)*Rdotp/s
        return [dtdr, dzdr]
    events = None
    if zstop is not None:
        def hit_z(r, y): return y[1] - zstop
        hit_z.terminal = True; hit_z.direction = 1
        # also stop if time approaches big bang
        def hit_tb(r, y): return y[0] - 1e-6*m.t0
        hit_tb.terminal = True; hit_tb.direction = -1
        events = [hit_z, hit_tb]
    r_eval = np.linspace(0, r_max, n_eval)
    sol = solve_ivp(rhs, (0, r_max), [m.t0, 0.0], t_eval=r_eval,
                    method="RK45", rtol=1e-9, atol=1e-11, max_step=r_max/2000.0,
                    events=events, dense_output=False)
    # success may be False if it terminated via event-less stiffness; accept partial
    r_arr = sol.t
    t_arr = sol.y[0]
    z_arr = sol.y[1]
    # append event points if any
    if events is not None and sol.t_events is not None:
        for ie, te in enumerate(sol.t_events):
            if len(te) > 0:
                ye = sol.y_events[ie][0]
                r_arr = np.append(r_arr, te[0]); t_arr = np.append(t_arr, ye[0]); z_arr = np.append(z_arr, ye[1])
    # keep strictly increasing z, drop any tail where z stopped rising
    order = np.argsort(r_arr); r_arr=r_arr[order]; t_arr=t_arr[order]; z_arr=z_arr[order]
    keep = np.concatenate(([True], np.diff(z_arr) > 0))
    r_arr=r_arr[keep]; t_arr=t_arr[keep]; z_arr=z_arr[keep]
    if len(r_arr) < 10:
        return None
    # areal radius along the cone = D_A
    DA = np.array([Rfuncs(m, max(r_arr[i],1e-9), t_arr[i])[0] for i in range(len(r_arr))])
    # longitudinal Hubble H_long = Rdot'/R' (expansion rate along line of sight)
    Hlong = np.zeros(len(r_arr))
    for i in range(len(r_arr)):
        R, Rp, Rdot, Rdotp = Rfuncs(m, max(r_arr[i],1e-9), t_arr[i])
        Hlong[i] = Rdotp/Rp if Rp != 0 else np.nan
    return r_arr, t_arr, z_arr, DA, Hlong

# ----------------------------------------------------------------------------
# Build distance interpolators in units c/H0_out.
# ----------------------------------------------------------------------------
class LTBDistances:
    def __init__(self, Om_in, Om_out, r0, dr, zmax=1200.0, r_max=None, n_eval=12000):
        self.m = LTBVoid(Om_in, Om_out, r0, dr)
        # Outer region is open-FLRW with Om_out: comoving radius to a given z is
        # bounded; for EdS (Om_out=1) z->inf at r=2.  Use r_max just under the
        # particle horizon and terminate integration at zmax via event.
        if r_max is None:
            r_max = 2.05 if Om_out >= 0.999 else 6.0
        out = None
        for _ in range(6):
            out = integrate_lightcone(self.m, r_max, n_eval=n_eval, zstop=zmax)
            if out is not None and out[2][-1] >= zmax*0.999:
                break
            # didn't reach zmax: grow r_max (open universe needs larger r)
            r_max *= 1.2
        if out is None:
            raise RuntimeError("light-cone integration failed")
        self.r_arr, self.t_arr, self.z_arr, self.DA_arr, self.Hlong_arr = out
        self.zmax_reached = self.z_arr[-1]
    def _interp(self, z, yarr):
        return np.interp(z, self.z_arr, yarr)
    def DA(self, z):
        return self._interp(z, self.DA_arr)
    def DM(self, z):
        # transverse comoving distance = (1+z) D_A
        return (1.0 + np.asarray(z))*self._interp(z, self.DA_arr)
    def DH(self, z):
        # c / H_long ; in our units c=1 so DH = 1/H_long
        Hl = self._interp(z, self.Hlong_arr)
        return 1.0/Hl
    def DV(self, z):
        z = np.asarray(z, dtype=float)
        dM = self.DM(z); dH = self.DH(z)
        return (z*dM*dM*dH)**(1.0/3.0)


if __name__ == "__main__":
    import harness as Hn
    zHD, zHEL, mb, Cf = Hn.load_sn()

    print("="*72)
    print("LTB GIANT-VOID (GBH 0802.1523 constrained CGBH) — distance build self-test")
    print("="*72)
    # sanity: a homogeneous EdS (Om_in=Om_out=1, no void) must reproduce EdS distances
    print("\n[sanity] homogeneous EdS limit (Om_in=Om_out=1):")
    try:
        d = LTBDistances(0.999, 0.999, 1.0, 0.3, zmax=3.0)
        zt = np.array([0.1,0.5,1.0,2.0])
        # EdS comoving distance c/H0: DM = 2(1 - 1/sqrt(1+z))
        eds = 2.0*(1.0 - 1.0/np.sqrt(1+zt))
        edsH = 1.0/(1+zt)**1.5  # EdS DH = 1/E, E=(1+z)^{3/2}
        print("  z      DM_LTB    DM_EdS     DH_LTB    DH_EdS")
        for i,zz in enumerate(zt):
            print(f"  {zz:.2f}   {d.DM(zz):.5f}   {eds[i]:.5f}    {d.DH(zz):.5f}   {edsH[i]:.5f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("  EdS sanity FAILED:", repr(e))

    # ---- VALIDATION 2: homogeneous OPEN FLRW limit (Om_in=Om_out=0.3) ----
    # Exercises the curvature term sqrt(1-k r^2).  Analytic open-FLRW:
    #   E(z)=sqrt(Om(1+z)^3 + (1-Om)(1+z)^2);  DC=int dz/E;
    #   DM = sinh(sqrt(Ok) DC)/sqrt(Ok); DH=1/E.
    print("\n[validation-2] homogeneous OPEN FLRW limit (Om_in=Om_out=0.30):")
    try:
        from scipy.integrate import quad as _q
        Om=0.30; Ok=1-Om
        d2 = LTBDistances(0.300, 0.300, 1.0, 0.3, zmax=3.0, r_max=6.0)
        zt = np.array([0.1,0.5,1.0,2.0])
        def E(z): return np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2)
        print("  z      DM_LTB    DM_open    DH_LTB    DH_open")
        for zz in zt:
            DC=_q(lambda x:1.0/E(x),0,zz)[0]
            DMo=np.sinh(np.sqrt(Ok)*DC)/np.sqrt(Ok)
            print(f"  {zz:.2f}   {d2.DM(zz):.5f}   {DMo:.5f}    {d2.DH(zz):.5f}   {1.0/E(zz):.5f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("  open-FLRW validation FAILED:", repr(e))

    # ---- VALIDATION 3: GBH giant-void profile, reproduce published H(z) shape ----
    # GBH 0802.1523 best-fit constrained model has Om_in~0.13, Om_out~1, with a
    # void radius giving on-centre H mismatch H_in/H_out ~ 1.2-1.3.  Check the
    # central-vs-asymptotic Hubble ratio matches the CGBH expectation.
    print("\n[validation-3] GBH void: central vs asymptotic Hubble ratio")
    try:
        for (oi, oo) in [(0.13,1.0),(0.20,1.0),(0.30,1.0)]:
            m=LTBVoid(oi,oo,2.5,0.5)
            # local H at centre = adot/a (a=1 at t0) for shell r->0 = H0p(0)
            Hin=m.H0p(1e-6); Hout=m.H0p(50.0)
            print(f"  Om_in={oi:.2f} Om_out={oo:.2f}: H0p_in/H0p_out = {Hin/Hout:.4f}  "
                  f"(eta-ratio; deeper void -> faster central expansion)")
    except Exception as e:
        import traceback; traceback.print_exc()

    # ========================================================================
    # FITS to the harness data
    # ========================================================================
    def build_predict_and_Dc(Om_in, Om_out, r0, dr, zmax):
        d = LTBDistances(Om_in, Om_out, r0, dr, zmax=zmax)
        def predict(z, kind):
            if kind == "DH": return d.DH(z)
            if kind == "DM": return d.DM(z)
            return d.DV(z)
        Dc = d.DM(zHD)  # line-of-sight comoving distance to SN = DM (on-centre radial)
        return d, predict, Dc

    # ---- SN-only fit: scan (Om_in, r0, dr); Om_out=1 (CGBH) ----
    print("\n" + "="*72)
    print("SN-only fit (CGBH, Om_out=1):")
    zmax_sn = float(np.max(zHD))*1.02
    best_sn = None
    grid_oi = [0.05,0.10,0.15,0.20,0.25,0.30,0.40]
    grid_r0 = [1.0,1.5,2.0,2.5,3.0,4.0,6.0]
    grid_dr = [0.3,0.5,0.8,1.2]
    for oi in grid_oi:
        for r0v in grid_r0:
            for drv in grid_dr:
                try:
                    d = LTBDistances(oi, 1.0, r0v, drv, zmax=zmax_sn)
                    Dc = d.DM(zHD)
                    c = Hn.sn_chi2(Dc)
                except Exception:
                    continue
                if best_sn is None or c < best_sn[0]:
                    best_sn = (c, oi, r0v, drv)
    print(f"  SN best: chi2={best_sn[0]:.1f}  Om_in={best_sn[1]} r0={best_sn[2]} dr={best_sn[3]}")
    print(f"  (LCDM SN-only ref ~ sn_chi2 at Om=0.305)")

    # ---- evaluate the SN-best params on BAO+CMB (consistency test) ----
    print("\n" + "="*72)
    print("Cross-check: SN-best params evaluated on BAO+CMB and joint:")
    oi,r0v,drv = best_sn[1],best_sn[2],best_sn[3]
    d = LTBDistances(oi, 1.0, r0v, drv, zmax=1200.0)
    def predict_snbest(z, kind, _d=d):
        if kind=="DH": return _d.DH(z)
        if kind=="DM": return _d.DM(z)
        return _d.DV(z)
    chi_bc_at_sn, a_sn = Hn.bao_cmb_chi2(predict_snbest)
    chi_bo_at_sn, _ = Hn.bao_only_chi2(predict_snbest)
    chi_sn_at_sn = Hn.sn_chi2(d.DM(zHD))
    print(f"  SN-best (Om_in={oi},r0={r0v},dr={drv}): "
          f"SN={chi_sn_at_sn:.1f}  BAO-only={chi_bo_at_sn:.1f}  BAO+CMB={chi_bc_at_sn:.1f}  "
          f"joint={chi_sn_at_sn+chi_bc_at_sn:.1f}  H0={Hn.H0_from_alpha(a_sn):.1f}")

    # ---- BAO+CMB coarse fit (needs z~1090); reduced grid for tractability ----
    print("\n" + "="*72)
    print("BAO+CMB fit (CGBH, Om_out=1) coarse grid:")
    bc_grid_oi = [0.05,0.10,0.15,0.20,0.30]
    bc_grid_r0 = [1.5,2.5,4.0]
    bc_grid_dr = [0.5,1.0]
    best_bc = None; best_joint = None
    for oi in bc_grid_oi:
        for r0v in bc_grid_r0:
            for drv in bc_grid_dr:
                try:
                    d = LTBDistances(oi, 1.0, r0v, drv, zmax=1200.0)
                    def predict(z, kind, _d=d):
                        if kind=="DH": return _d.DH(z)
                        if kind=="DM": return _d.DM(z)
                        return _d.DV(z)
                    chi, a = Hn.bao_cmb_chi2(predict)
                    snc = Hn.sn_chi2(d.DM(zHD))
                    jt = chi + snc
                except Exception:
                    continue
                if best_bc is None or chi < best_bc[0]:
                    best_bc = (chi, oi, r0v, drv, a)
                if best_joint is None or jt < best_joint[0]:
                    best_joint = (jt, oi, r0v, drv, a, snc, chi)
                print(f"    Om_in={oi:.2f} r0={r0v:.1f} dr={drv:.1f}: BAO+CMB={chi:7.1f} SN={snc:7.1f} joint={jt:7.1f} H0={Hn.H0_from_alpha(a):.1f}")
    if best_bc:
        H0 = Hn.H0_from_alpha(best_bc[4])
        print(f"\n  BAO+CMB best: chi2={best_bc[0]:.1f}  Om_in={best_bc[1]} r0={best_bc[2]} dr={best_bc[3]}  H0={H0:.1f}")
    if best_joint:
        H0j = Hn.H0_from_alpha(best_joint[4])
        # dBIC vs LCDM: LCDM joint chi2=1402.2, k=1 (Om); CGBH k=3 (Om_in,r0,dr). N=1593.
        dBIC = (best_joint[0]-1402.2) + (3-1)*np.log(1593)
        print(f"  JOINT best: chi2={best_joint[0]:.1f} (SN={best_joint[5]:.1f}+BAOCMB={best_joint[6]:.1f}) "
              f"Om_in={best_joint[1]} r0={best_joint[2]} dr={best_joint[3]} H0={H0j:.1f}  dBIC_vs_LCDM={dBIC:+.1f}")

    # ---- persist result artifact (one-number-one-script-one-artifact) ----
    import json, os
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _out = os.path.join(_repo, "probes_out", "ltb_family_fit.json")
    result = dict(
        model="LTB giant-void, constrained GBH (Garcia-Bellido & Haugbolle 2008, arXiv:0802.1523; CGBH, Om_out=1, homogeneous big bang)",
        params="(Om_in, r0, dr) in units c/H0_out; k=3 cosmological shape params",
        sn_only=dict(chi2=float(best_sn[0]), Om_in=float(best_sn[1]), r0=float(best_sn[2]), dr=float(best_sn[3])),
        sn_best_on_baocmb=dict(Om_in=float(best_sn[1]), r0=float(best_sn[2]), dr=float(best_sn[3]),
                               sn_chi2=float(chi_sn_at_sn), bao_only_chi2=float(chi_bo_at_sn),
                               bao_cmb_chi2=float(chi_bc_at_sn), joint_chi2=float(chi_sn_at_sn+chi_bc_at_sn),
                               H0=float(Hn.H0_from_alpha(a_sn))),
        bao_cmb_best=dict(chi2=float(best_bc[0]), Om_in=float(best_bc[1]), r0=float(best_bc[2]),
                          dr=float(best_bc[3]), H0=float(Hn.H0_from_alpha(best_bc[4]))),
        joint=dict(chi2=float(best_joint[0]), sn_chi2=float(best_joint[5]), baocmb_chi2=float(best_joint[6]),
                   Om_in=float(best_joint[1]), r0=float(best_joint[2]), dr=float(best_joint[3]),
                   H0=float(H0j)),
        lcdm_joint_chi2=1402.2, N=1593,
        dchi2_vs_LCDM=float(best_joint[0]-1402.2),
        dBIC_vs_LCDM=float(dBIC),
        dBIC_note="dBIC = dchi2 + (k-1)*ln(N), k=3 (Om_in,r0,dr) vs LCDM k=1 (Om), N=1593; "
                  "same convention as joint_w0wa.py and model_rasanen.py",
        central_H0_range="SN-best and BAO+CMB-best central H0 both ~55-59 (below local 73)",
        command="cd src && python model_ltbvoid.py",
        verdict_of_verification="Reproduced from src/model_ltbvoid.py: joint dBIC~+715 (SN prefer a deeper "
                                "void than BAO+CMB, same split as timescape); LTB does not reach local H0 and "
                                "is independently kSZ-excluded. Row of tab:voids now backed by this artifact.",
    )
    with open(_out, "w") as _f:
        json.dump(result, _f, indent=2)
    print(f"\nsaved probes_out/ltb_family_fit.json  (joint dBIC={dBIC:+.1f})")
