#!/usr/bin/env python3
# freshH0: feasibility probe for the paper's "what a decisive test requires (ii)" -- replace the
# 2008-era literature late-time dressed H0 (61.7 +1.2/-1.1, Leith+2008) with a FRESH SH0ES-style
# measurement from Pantheon+SH0ES: Cepheid calibrators (CEPH_DIST, IS_CALIBRATOR==1) anchor M_B,
# Hubble-flow SNe constrain (H0, shape); flat LCDM fits (H0, M_B, Om), timescape fits
# (H0_dressed, M_B, fv0) with Hbar0 = H0_dressed*2(2+fv0)/(4fv0^2+fv0+4). Full 1701x1701 stat+sys
# covariance, GLS-analytic profiling of (M_B, 5log10 H0), grid profile over the shape parameter.
# Outputs: fresh dressed H0_TS +- err, fv0, new tension T vs the timescape CMB value
# 61.0 (+-0.79 stat, +-4.88 sys), the LCDM analogue vs Planck 67.36+-0.54, and the updated
# delta_req = H0_fresh/H0_global - 1 against the predicted 17-22% void-variance window
# (paper claim A, tex lines 158-183). Writes probes_out/freshH0.json.
import os, sys, json, hashlib, time
import numpy as np

WORKTREE = "/Users/s/dev/science/timescape-hubble-tension/.claude/worktrees/significance-audit"
SRC = os.path.join(WORKTREE, "src")
os.chdir(SRC)
sys.path.insert(0, SRC)

import fit_timescape as F                      # reuse repo distance machinery, unmodified
from scipy.linalg import cho_factor, cho_solve
import pandas as pd

C_KMS = 299792.458
H0REF = 70.0                                   # reference H0 for the fiducial mu; q rescales it
DATA  = "data/PantheonSH0ES.dat"
COV   = "data/PantheonSH0ES_STATSYS.cov"
OUT   = os.path.join(WORKTREE, "probes_out", "freshH0.json")

# ---- paper claim-A anchors (tex:158-183) ----
H0_TS_CMB, SIG_CMB_STAT, SIG_CMB_SYS = 61.0, 0.79, 4.88
H0_TS_SN2008, SIG_SN2008 = 61.7, 1.15          # +1.2/-1.1 symmetrised
H0_PLANCK, SIG_PLANCK = 67.36, 0.54
H0_SH0ES,  SIG_SH0ES  = 73.04, 1.04
WINDOW = (0.17, 0.22)                          # predicted apparent-Hubble variance window

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ----------------------------------------------------------------------
# distances (absolute, Mpc)
# ----------------------------------------------------------------------
def hbar0_of_dressed(H0d, fv0):
    return H0d * 2.0*(2.0+fv0)/(4.0*fv0**2 + fv0 + 4.0)

def mu_TS(zhd, zhel, fv0, H0d):
    # repo convention: D_comoving = (1+zHD)*dA (units c/Hbar0); d_L = (1+zHEL)*D
    Dsh = F.D_shape_TS(zhd, fv0)
    dL = (C_KMS/hbar0_of_dressed(H0d, fv0)) * (1.0+zhel) * Dsh
    return 5.0*np.log10(dL) + 25.0

def mu_L(zhd, zhel, Om, H0):
    Dsh = F.D_shape_LCDM(zhd, Om)
    dL = (C_KMS/H0) * (1.0+zhel) * Dsh
    return 5.0*np.log10(dL) + 25.0

# ----------------------------------------------------------------------
# absolute-normalisation gates: lim_{z->0} d_L * H0_dressed / (c z) -> 1
# (the dressed H0 is DEFINED as the low-z Hubble-law coefficient, so this
#  validates that the repo shape functions carry no fv0-dependent constant
#  that would bias a calibrator-anchored H0)
# ----------------------------------------------------------------------
def norm_checks():
    zs = np.array([1e-3, 5e-4, 2e-4, 1e-4])
    fv0, H0d = 0.70, 61.7
    dl = (C_KMS/hbar0_of_dressed(H0d, fv0)) * (1.0+zs) * F.D_shape_TS(zs, fv0)
    a_ts = float(np.polyfit(zs, dl*H0d/(C_KMS*zs), 1)[1])
    Om, H0 = 0.30, 70.0
    dl = (C_KMS/H0) * (1.0+zs) * F.D_shape_LCDM(zs, Om)
    a_l = float(np.polyfit(zs, dl*H0/(C_KMS*zs), 1)[1])
    return a_ts, a_l

# ----------------------------------------------------------------------
# GLS: r = y - MB*1 - q*w, q = -5 log10(H0/H0REF)  (profiled analytically)
# ----------------------------------------------------------------------
def run_model(cf, w, mb_sel, mu0_of_shape, grid):
    n = len(mb_sel)
    A = np.column_stack([np.ones(n), w.astype(float)])
    CiA = cho_solve(cf, A)
    M = A.T @ CiA
    Minv = np.linalg.inv(M)
    chi2_g = np.empty(len(grid)); MB_g = np.empty(len(grid)); q_g = np.empty(len(grid))
    for k, s in enumerate(grid):
        y = mb_sel - mu0_of_shape(s)
        Ciy = cho_solve(cf, y)
        v = A.T @ Ciy
        th = Minv @ v
        chi2_g[k] = y @ Ciy - v @ th
        MB_g[k], q_g[k] = th
    return chi2_g, MB_g, q_g, Minv

def package(grid, chi2_g, MB_g, q_g, Minv, n, label):
    i = int(np.argmin(chi2_g)); cmin = float(chi2_g[i])
    railed = bool(i == 0 or i == len(grid)-1)
    d = chi2_g - cmin
    sh_lo = float(grid[i]-np.interp(1.0, d[:i+1][::-1], grid[:i+1][::-1])) if (i > 0 and d[:i+1].max() >= 1.0) else float("nan")
    sh_hi = float(np.interp(1.0, d[i:], grid[i:])-grid[i]) if (i < len(grid)-1 and d[i:].max() >= 1.0) else float("nan")
    sq2 = float(Minv[1, 1])
    span = 12.0*np.sqrt(sq2)
    for _ in range(6):
        qg = np.linspace(q_g[i]-span, q_g[i]+span, 4001)
        prof = np.min(chi2_g[None, :] + (qg[:, None]-q_g[None, :])**2/sq2, axis=1)
        j = int(np.argmin(prof)); dprof = prof - prof[j]
        if dprof[0] >= 1.0 and dprof[-1] >= 1.0 and 0 < j < len(qg)-1:
            break
        span *= 2.0
    qhat = float(qg[j])
    qlo = float(np.interp(1.0, dprof[:j+1][::-1], qg[:j+1][::-1]))
    qhi = float(np.interp(1.0, dprof[j:], qg[j:]))
    H0 = H0REF*10**(-qhat/5.0)
    H0_hi = H0REF*10**(-qlo/5.0) - H0            # smaller q -> larger H0
    H0_lo = H0 - H0REF*10**(-qhi/5.0)
    sigH0 = 0.5*(H0_hi+H0_lo)
    sigH0_cond = H0*np.log(10.0)/5.0*np.sqrt(sq2)  # shape fixed at best (Fisher)
    res = dict(label=label, n=n, dof=n-3,
               shape_best=float(grid[i]), shape_err_lo=sh_lo, shape_err_hi=sh_hi,
               shape_railed=railed,
               H0=float(H0), H0_err_lo=float(H0_lo), H0_err_hi=float(H0_hi),
               H0_err_sym=float(sigH0), H0_err_fixed_shape=float(sigH0_cond),
               MB=float(MB_g[i]), MB_err_cond=float(np.sqrt(Minv[0, 0])),
               chi2_min=cmin, chi2_per_dof=cmin/(n-3),
               grid=[float(grid[0]), float(grid[-1]), len(grid)])
    log(f"  {label}: shape={res['shape_best']:.4f} (+{sh_hi:.4f}/-{sh_lo:.4f}) "
        f"H0={H0:.2f} (+{H0_hi:.2f}/-{H0_lo:.2f}) MB={res['MB']:.3f} "
        f"chi2={cmin:.1f}/{n-3} railed={railed}")
    return res

# ----------------------------------------------------------------------
def clean(o):
    """Replace non-finite floats with None so the output is strict JSON."""
    if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return None
    return o

# ----------------------------------------------------------------------
def main():
    log("loading data...")
    df = pd.read_csv(DATA, sep=r"\s+")
    n_all = len(df)
    with open(COV) as f:
        n_cov = int(f.readline())
    assert n_cov == n_all, f"cov dim {n_cov} != dat rows {n_all}"
    vals = np.fromfile(COV, sep=" ")
    assert vals.size == n_cov*n_cov + 1
    Cfull = vals[1:].reshape(n_cov, n_cov)
    md5_dat = hashlib.md5(open(DATA, "rb").read()).hexdigest()
    md5_cov = hashlib.md5(open(COV, "rb").read()).hexdigest()
    log(f"N={n_all} rows, cov {n_cov}x{n_cov}")

    zHD = df["zHD"].to_numpy(float); zHEL = df["zHEL"].to_numpy(float)
    mb = df["m_b_corr"].to_numpy(float); ceph = df["CEPH_DIST"].to_numpy(float)
    iscal = df["IS_CALIBRATOR"].to_numpy(float).astype(int) == 1
    used_hf = df["USED_IN_SH0ES_HF"].to_numpy(float).astype(int) == 1

    a_ts, a_l = norm_checks()
    log(f"norm gates: TS low-z d_L*H0d/(cz)->{a_ts:.6f}  LCDM ->{a_l:.6f}")
    assert abs(a_l - 1.0) < 1e-4, "LCDM absolute normalisation broken"
    assert abs(a_ts - 1.0) < 1e-3, "TS absolute normalisation broken (fv0-dep constant?)"

    fv_grid = np.arange(0.35, 0.9500001, 0.0025)
    om_grid = np.arange(0.05, 0.6000001, 0.0025)

    variants = {
        "main_z001":     (~iscal) & (zHD > 0.01),               # repo cosmology cut
        "shoes_hf_flag": (~iscal) & used_hf,                    # Riess-style 0.0233<z<0.15 sample
        "hf_z_gt_010":   (~iscal) & (zHD > 0.10),               # beyond any void/homogeneity scale
    }
    results = {}
    for vname, hf in variants.items():
        sel = iscal | hf
        idx = np.where(sel)[0]
        w = hf[idx.copy()]                       # HF indicator within selection
        # order: keep original row order; calibrators have w=0 and mu0=CEPH_DIST
        zhd_s, zhel_s, mb_s, ceph_s = zHD[idx], zHEL[idx], mb[idx], ceph[idx]
        w_s = hf[idx]
        cal_s = iscal[idx]
        n_sel = len(idx)
        log(f"variant {vname}: n={n_sel} (cal={int(cal_s.sum())}, HF={int(w_s.sum())}), "
            f"HF z range {zhd_s[w_s].min():.4f}..{zhd_s[w_s].max():.4f}; Cholesky...")
        cf = cho_factor(Cfull[np.ix_(sel, sel)])

        mu0 = np.where(cal_s, ceph_s, 0.0)
        zhd_hf, zhel_hf = zhd_s[w_s], zhel_s[w_s]

        def mu0_TS(fv0):
            m = mu0.copy(); m[w_s] = mu_TS(zhd_hf, zhel_hf, fv0, H0REF); return m
        def mu0_L(Om):
            m = mu0.copy(); m[w_s] = mu_L(zhd_hf, zhel_hf, Om, H0REF); return m

        rL = package(om_grid, *run_model(cf, w_s, mb_s, mu0_L, om_grid), n_sel, f"{vname}/LCDM")
        rT = package(fv_grid, *run_model(cf, w_s, mb_s, mu0_TS, fv_grid), n_sel, f"{vname}/timescape")
        rT["Om_dressed_equiv"] = 0.5*(1.0-rT["shape_best"])*(2.0+rT["shape_best"])
        rT["Hbar0"] = hbar0_of_dressed(rT["H0"], rT["shape_best"])
        results[vname] = dict(n=n_sel, n_cal=int(cal_s.sum()), n_hf=int(w_s.sum()),
                              lcdm=rL, timescape=rT,
                              dchi2_TS_minus_LCDM=rT["chi2_min"]-rL["chi2_min"])

    # ---- validation gate: LCDM main fit must recover ~73.0 +- 1.0 ----
    H0L, sL = results["main_z001"]["lcdm"]["H0"], results["main_z001"]["lcdm"]["H0_err_sym"]
    gate = (71.5 <= H0L <= 75.0) and (0.7 <= sL <= 1.5)
    log(f"GATE LCDM main: H0={H0L:.2f}+-{sL:.2f} -> {'PASS' if gate else 'FAIL'}")
    if not gate:
        json.dump(clean(dict(gate_failed=True, results=results)), open(OUT, "w"),
                  indent=1, allow_nan=False)
        sys.exit("LCDM validation gate failed -- debug before interpreting")

    # ---- tension statistics with the FRESH numbers (main variant) ----
    def T(a, sa, b, sb): return abs(a-b)/np.hypot(sa, sb)
    H0T = results["main_z001"]["timescape"]["H0"]
    sT = results["main_z001"]["timescape"]["H0_err_sym"]
    H0T10 = results["hf_z_gt_010"]["timescape"]["H0"]
    sT10 = results["hf_z_gt_010"]["timescape"]["H0_err_sym"]
    sig_cmb_tot = float(np.hypot(SIG_CMB_STAT, SIG_CMB_SYS))
    tension = dict(
        fresh_TS_vs_CMB61p0_stat=float(T(H0T, sT, H0_TS_CMB, SIG_CMB_STAT)),
        fresh_TS_vs_CMB61p0_statsys=float(T(H0T, sT, H0_TS_CMB, sig_cmb_tot)),
        fresh_TS_vs_SN2008_61p7=float(T(H0T, sT, H0_TS_SN2008, SIG_SN2008)),
        fresh_LCDM_vs_Planck=float(T(H0L, sL, H0_PLANCK, SIG_PLANCK)),
        fresh_TS_zgt010_vs_CMB61p0_stat=float(T(H0T10, sT10, H0_TS_CMB, SIG_CMB_STAT)),
        fresh_TS_zgt010_vs_CMB61p0_statsys=float(T(H0T10, sT10, H0_TS_CMB, sig_cmb_tot)),
        paper_claimA_TS_stat=0.50, paper_claimA_TS_statsys=0.14,
        paper_LCDM_reference=4.85,
        shoes_published_vs_Planck=float(T(H0_SH0ES, SIG_SH0ES, H0_PLANCK, SIG_PLANCK)))

    # ---- delta_req: fractional excess needed to reconcile the ladder-anchored
    #      dressed H0 with the global dressed value (paper: 73.04/61.7-1=0.184+-0.028,
    #      window 17-22%) ----
    def dreq(a, sa, b, sb):
        d = a/b - 1.0
        sd = float(np.hypot(sa/b, a*sb/b**2))
        return dict(delta=float(d), sigma=sd,
                    in_window=bool(WINDOW[0] <= d <= WINDOW[1]),
                    nsig_below_17=float((WINDOW[0]-d)/sd), nsig_above_22=float((d-WINDOW[1])/sd))
    delta_req = dict(
        paper_ref_73p04_over_61p7=dreq(H0_SH0ES, SIG_SH0ES, H0_TS_SN2008, SIG_SN2008),
        fresh_over_CMB61p0_stat=dreq(H0T, sT, H0_TS_CMB, SIG_CMB_STAT),
        fresh_over_CMB61p0_statsys=dreq(H0T, sT, H0_TS_CMB, sig_cmb_tot),
        fresh_over_SN2008_61p7=dreq(H0T, sT, H0_TS_SN2008, SIG_SN2008),
        fresh_zgt010_over_CMB61p0_stat=dreq(H0T10, sT10, H0_TS_CMB, SIG_CMB_STAT),
        window=list(WINDOW),
        ladder_H0_range_implied_by_window_on_61p0=[61.0*1.17, 61.0*1.22],
        note_zgt010="hf_z_gt_010 restricts Hubble-flow SNe to z>0.10 (>~300/h Mpc), entirely "
                    "beyond the ~100/h Mpc homogeneity scale and ~30/h Mpc dominant-void scale "
                    "where the 17-22% apparent-Hubble variance is predicted to act; the "
                    "calibrator rung (geometric Cepheid distances, M_B) involves no redshift, "
                    "so a still-high dressed H0 here cannot be attributed to Hubble-flow "
                    "expansion variance")

    out = dict(
        probe="freshH0",
        purpose="fresh SH0ES-style late-time dressed H0 for timescape and LCDM from Pantheon+SH0ES "
                "(calibrators anchor M_B; full stat+sys covariance; (MB,q) GLS-profiled; shape gridded)",
        data=dict(dat_rows=n_all, cov_dim=n_cov, md5_dat=md5_dat, md5_cov=md5_cov,
                  n_calibrators=int(iscal.sum())),
        conventions=dict(H0REF=H0REF, q_def="mu = mu(H0REF) + q*w_HF, H0 = H0REF*10^(-q/5)",
                         calibrator_residual="m_b_corr - M_B - CEPH_DIST",
                         hf_residual="m_b_corr - M_B - mu_model(zHD,zHEL;shape,H0)",
                         timescape_scale="Hbar0 = H0_dressed*2(2+fv0)/(4fv0^2+fv0+4); "
                                         "d_L=(c/Hbar0)(1+zHEL)(1+zHD)dA"),
        norm_gates=dict(TS_lowz_limit=a_ts, LCDM_lowz_limit=a_l),
        lcdm_validation_gate=dict(passed=gate, H0=H0L, sigma=sL, expected="~73.0 +- 1.0"),
        variants=results, tension=tension, delta_req=delta_req)
    json.dump(clean(out), open(OUT, "w"), indent=1, allow_nan=False)
    log(f"wrote {OUT}")

if __name__ == "__main__":
    main()
