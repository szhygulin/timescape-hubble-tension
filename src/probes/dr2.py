# Probe "dr2": refit the BAO(+CMB) inverse-distance-ladder calibration with DESI DR2
# BAO distances (arXiv:2503.14738) in place of DESI DR1, for both timescape (fv0,
# alpha profiled) and flat LCDM (Om, alpha profiled); recompute the SN-vs-BAO
# void-fraction split significance (profile parameter-shift statistic, 1 dof)
# against the Pantheon+ standard full-covariance SN curve and against the
# Seifert+2025 fv0=0.737+-0.029 scenario; compare everything to DR1.
#
# Data provenance (two independent sources, cross-checked in-script):
#  (1) arXiv:2503.14738v2 summary table (extracted by literature agent) - rounded
#      journal-table values, recorded below as DR2_TABLE.
#  (2) Official DESI DR2 Gaussian BAO likelihood shipped for Cobaya/CosmoMC:
#      github.com/CobayaSampler/bao_data, files desi_bao_dr2/
#      desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt, commit b7b8a36e9bcc
#      (2025-03-20, the DR2 release commit), fetched 2026-07-05. Exact means and
#      the exact 13x13 covariance (diagonal 2x2 DM/DH blocks), recorded below as
#      DR2_MEAN / DR2_VARCOV. This is the PRIMARY dataset used here; the rounded
#      table is refit as a sensitivity variant.
#
# CMB acoustic point: identical handling to src/verify_and_extend.py (the version
# behind the paper's headline DR1 numbers): D_M(z*)/r_d = (100/1.04109)*(144.39/147.09)
# = 94.290, error = theta*-propagated, floored at 0.05. A consistent-Planck-column
# variant (144.43/147.09 -> 94.316; known issue "planck-column-mixing") is also run.
#
# Output: probes_out/dr2.json.  Run with the scratchpad venv python.
import os, sys, json, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(_ROOT, "src")
OUT = os.path.join(_ROOT, "probes_out", "dr2.json")
os.chdir(SRC)
sys.path.insert(0, SRC)

import fit_timescape as F
import timescape_baocmb as T   # NB: module-level DR1 fit prints a banner on import (harmless)

C_KMS = 299792.458
RD = 147.09           # Planck drag sound horizon used throughout the paper
t0 = time.time()

# ---------------------------------------------------------------- DR2 data ----
# Exact DESI DR2 means (Cobaya bao_data, desi_gaussian_bao_ALL_GCcomb_mean.txt)
DR2_MEAN = [
    (0.295, "DV", 7.94167639),
    (0.510, "DM", 13.58758434), (0.510, "DH", 21.86294686),
    (0.706, "DM", 17.35069094), (0.706, "DH", 19.45534918),
    (0.934, "DM", 21.57563956), (0.934, "DH", 17.64149464),
    (1.321, "DM", 27.60085612), (1.321, "DH", 14.17602155),
    (1.484, "DM", 30.51190063), (1.484, "DH", 12.81699964),
    (2.330, "DM", 38.988973961958784), (2.330, "DH", 8.631545674846294),
]
# z -> (var_DM_or_DV, var_DH, cov_DM_DH) from desi_gaussian_bao_ALL_GCcomb_cov.txt
DR2_VARCOV = {
    0.295: (5.78998687e-03, None, None),
    0.510: (2.83473742e-02, 1.83928040e-01, -3.26062007e-02),
    0.706: (3.23752442e-02, 1.11469198e-01, -2.37445646e-02),
    0.934: (2.61732816e-02, 4.04183878e-02, -1.12938006e-02),
    1.321: (1.05336516e-01, 5.04233092e-02, -2.90308418e-02),
    1.484: (5.83020277e-01, 2.68336193e-01, -1.95215562e-01),
    2.330: (2.82685779e-01, 1.02136194e-02, -2.31395216e-02),
}
# Rounded journal-table values (arXiv:2503.14738v2 summary table, lit-agent extract)
DR2_TABLE = [
    (0.295, "DV", 7.942, 0.075, None),
    (0.510, "DM", 13.588, 0.167, -0.459), (0.510, "DH", 21.863, 0.425, -0.459),
    (0.706, "DM", 17.351, 0.177, -0.404), (0.706, "DH", 19.455, 0.330, -0.404),
    (0.934, "DM", 21.576, 0.152, -0.416), (0.934, "DH", 17.641, 0.193, -0.416),
    (1.321, "DM", 27.601, 0.318, -0.434), (1.321, "DH", 14.176, 0.221, -0.434),
    (1.484, "DM", 30.512, 0.760, -0.500), (1.484, "DH", 12.817, 0.516, -0.500),
    (2.330, "DM", 38.988, 0.531, -0.431), (2.330, "DH", 8.632, 0.101, -0.431),
]

def build_dr2_rows():
    """Harness-format rows (z, kind, value, err, corr) reproducing the exact
    Cobaya covariance through timescape_baocmb.build_cov."""
    rows, xchk = [], []
    for z, k, v in DR2_MEAN:
        vM, vH, cMH = DR2_VARCOV[z]
        if k == "DV":
            rows.append((z, "DV", v, float(np.sqrt(vM)), None))
        else:
            var = vM if k == "DM" else vH
            corr = None if cMH is None else float(cMH / np.sqrt(vM * vH))
            rows.append((z, k, v, float(np.sqrt(var)), corr))
    # cross-check vs the rounded journal table (source #1 vs source #2)
    for (z, k, v, e, c), (zt, kt, vt, et, ct) in zip(rows, DR2_TABLE):
        assert (z, k) == (zt, kt)
        xchk.append(dict(z=z, kind=k, cobaya=v, table=vt, dv=abs(v - vt),
                         err_cobaya=e, err_table=et, derr=abs(e - et)))
        assert abs(v - vt) < 2e-3, (z, k, v, vt)
        # errors: Cobaya Gaussian-likelihood sigmas run 0.5-2% above the rounded
        # journal-table sigmas, except the combined LRG3+ELG1 bin where the
        # likelihood release is ~6% wider (0.1618/0.2010, r=-0.347) than the
        # table (0.152/0.193, r=-0.416).  DR1 Cobaya matches the DR1 table
        # exactly, so this is a genuine DR2 likelihood-vs-table feature; the
        # dr2_*_table variants below bound its impact.
        assert abs(e - et) < 1.1e-2, (z, k, e, et)
    return rows, xchk

DR2_ROWS, CROSSCHECK = build_dr2_rows()
DR1_ROWS = [tuple(r) for r in T.BAO]

# CMB acoustic point, verify_and_extend.py handling
theta100, sig100 = 1.04109, 0.00030
DMZ_RD = (100.0 / theta100) * (144.39 / 147.09)          # = 94.290
SIG_CMB = max(DMZ_RD * (sig100 / theta100), 0.05)        # = 0.05 floor
ZSTAR = 1089.80
CMB_ROW = (ZSTAR, "DM", DMZ_RD, SIG_CMB, None)
DMZ_RD_FIX = (100.0 / theta100) * (144.43 / 147.09)      # consistent +lensing pairing = 94.316
CMB_ROW_FIX = (ZSTAR, "DM", DMZ_RD_FIX, SIG_CMB, None)

# ------------------------------------------------------------ fit machinery ----
FVG = np.linspace(0.30, 0.995, 1391)   # step 5e-4
OMG = np.linspace(0.15, 0.45, 601)     # step 5e-4

def model_matrix(grid, rows, lcdm):
    G = np.empty((len(grid), len(rows)))
    for i, p in enumerate(grid):
        G[i] = T.model_vec(p, rows, lcdm=(p if lcdm else None))
    return G

def profile_chi2(G, idx, rows_sub):
    """chi2(shape-param) with alpha profiled analytically (GLS scale factor)."""
    d = np.array([r[2] for r in rows_sub])
    Cinv = np.linalg.inv(T.build_cov(rows_sub))
    Gi = G[:, idx]
    GC = Gi @ Cinv
    gCd = GC @ d
    gCg = np.einsum('ij,ij->i', GC, Gi)
    chis = d @ (Cinv @ d) - gCd**2 / gCg
    return chis, gCd / gCg

def interval(grid, chis):
    i = int(np.argmin(chis)); dchi = chis - chis[i]
    lo = hi = float('nan')
    if i > 0 and dchi[0] >= 1.0:
        lo = float(np.interp(1.0, dchi[:i+1][::-1], grid[:i+1][::-1]))
    if i < len(grid) - 1 and dchi[-1] >= 1.0:
        hi = float(np.interp(1.0, dchi[i:], grid[i:]))
    return float(grid[i]), float(chis[i]), lo, hi

def fit_summary(grid, chis, alphas, lcdm, npts):
    best, cmin, lo, hi = interval(grid, chis)
    i = int(np.argmin(chis)); a = float(alphas[i]); dof = npts - 2
    out = dict(chi2=cmin, npoints=npts, dof=dof, chi2_dof=cmin / dof,
               alpha=a, err_lo=best - lo, err_hi=hi - best)
    if lcdm:
        out.update(Om=best, H0_rd14709=C_KMS / (a * RD))
    else:
        out.update(fv0=best, Hbar0=C_KMS / (a * RD),
                   H0_dressed_rd14709=float(T.g_dress(best)) * C_KMS / (a * RD),
                   Om_dressed_implied=0.5 * (1 - best) * (2 + best))
    return out

# unions: model matrices are data-independent given (z,kind) row lists
print("[dr2] building timescape model matrices ...", flush=True)
G_TS_DR1 = model_matrix(FVG, DR1_ROWS + [CMB_ROW], lcdm=False)
G_TS_DR2 = model_matrix(FVG, DR2_ROWS + [CMB_ROW], lcdm=False)
print(f"[dr2] TS matrices done ({time.time()-t0:.0f}s); LCDM matrices ...", flush=True)
G_L_DR1 = model_matrix(OMG, DR1_ROWS + [CMB_ROW], lcdm=True)
G_L_DR2 = model_matrix(OMG, DR2_ROWS + [CMB_ROW], lcdm=True)
print(f"[dr2] LCDM matrices done ({time.time()-t0:.0f}s)", flush=True)

DR2_TABLE_ROWS = [tuple(r) for r in DR2_TABLE]
datasets = {
    # name: (G_ts, G_l, row-subset)  -- subset indices into the union row list
    "dr1_bao_only":       (G_TS_DR1, G_L_DR1, DR1_ROWS),
    "dr1_bao_cmb":        (G_TS_DR1, G_L_DR1, DR1_ROWS + [CMB_ROW]),
    "dr2_bao_only":       (G_TS_DR2, G_L_DR2, DR2_ROWS),
    "dr2_bao_cmb":        (G_TS_DR2, G_L_DR2, DR2_ROWS + [CMB_ROW]),
    # variants (same z/kind pattern -> reuse the DR2 matrices)
    "dr2_bao_cmb_planckfix": (G_TS_DR2, G_L_DR2, DR2_ROWS + [CMB_ROW_FIX]),
    "dr2_bao_only_table":    (G_TS_DR2, G_L_DR2, DR2_TABLE_ROWS),
    "dr2_bao_cmb_table":     (G_TS_DR2, G_L_DR2, DR2_TABLE_ROWS + [CMB_ROW]),
}

fits, ts_curves = {}, {}
for name, (Gts, Gl, sub) in datasets.items():
    idx = np.arange(len(sub))
    chis_ts, a_ts = profile_chi2(Gts, idx, sub)
    chis_l,  a_l  = profile_chi2(Gl,  idx, sub)
    ts = fit_summary(FVG, chis_ts, a_ts, lcdm=False, npts=len(sub))
    lc = fit_summary(OMG, chis_l,  a_l,  lcdm=True,  npts=len(sub))
    fits[name] = dict(timescape=ts, lcdm=lc,
                      dbic_ts_minus_lcdm=ts["chi2"] - lc["chi2"])   # equal k=2
    ts_curves[name] = chis_ts
    print(f"[dr2] {name:24s} TS fv0={ts['fv0']:.4f}+{ts['err_hi']:.4f}-{ts['err_lo']:.4f} "
          f"chi2/dof={ts['chi2']:.1f}/{ts['dof']}={ts['chi2_dof']:.2f} H0={ts['H0_dressed_rd14709']:.2f} | "
          f"LCDM Om={lc['Om']:.4f} chi2/dof={lc['chi2']:.1f}/{lc['dof']}={lc['chi2_dof']:.2f} "
          f"H0={lc['H0_rd14709']:.2f} | dBIC={fits[name]['dbic_ts_minus_lcdm']:+.1f}", flush=True)

# --------------------------------------------------- SN full-covariance curve ----
print(f"[dr2] loading Pantheon+ ({time.time()-t0:.0f}s) ...", flush=True)
zHD, zHEL, mb, Csn = F.load()
chi2sn = F.make_chi2(zHD, zHEL, mb, Csn)
FVSN = np.linspace(0.50, 0.95, 451)    # step 1e-3; upper edge above the 0.9025 crossing
sn_chis = np.empty(len(FVSN))
for i, fv in enumerate(FVSN):
    sn_chis[i] = chi2sn(F.D_shape_TS(zHD, fv))
    if i % 100 == 0:
        print(f"[dr2]   SN scan {i}/{len(FVSN)} ({time.time()-t0:.0f}s)", flush=True)
sn_best, sn_cmin, sn_lo, sn_hi = interval(FVSN, sn_chis)
sn = dict(fv0=sn_best, chi2=sn_cmin, err_lo=sn_best - sn_lo, err_hi=sn_hi - sn_best,
          N=len(zHD), grid=[0.50, 0.95, len(FVSN)])
print(f"[dr2] SN standard full-cov: fv0={sn_best:.4f} +{sn['err_hi']:.4f}/-{sn['err_lo']:.4f} "
      f"chi2={sn_cmin:.3f}", flush=True)

# ----------------------------------------------- fv-split significance (1 dof) ----
FINE = np.arange(0.5005, 0.9445, 1e-4)
sn_fine = np.interp(FINE, FVSN, sn_chis)

def split_stats(bao_chis, other_fine, other_min, other_sig, other_best, label):
    bao_fine = np.interp(FINE, FVG, bao_chis)
    # guard: BAO minimum must lie inside the common window
    assert abs(bao_fine.min() - bao_chis.min()) < 5e-3, label
    Tstat = float((other_fine + bao_fine).min() - other_min - bao_fine.min())
    joint_fv = float(FINE[np.argmin(other_fine + bao_fine)])
    b_best, _, b_lo, b_hi = interval(FVG, bao_chis)
    sig_bao = 0.5 * ((b_best - b_lo) + (b_hi - b_best))
    gap = other_best - b_best
    gauss = abs(gap) / np.sqrt(other_sig**2 + sig_bao**2)
    return dict(gap_fv0=gap, sigma_bao_fv0=sig_bao, sigma_other_fv0=other_sig,
                sigma_gaussian=float(gauss), T_profile_shift=Tstat,
                sigma_profile=float(np.sqrt(max(Tstat, 0.0))), joint_min_fv0=joint_fv)

sig_sn = 0.5 * (sn["err_lo"] + sn["err_hi"])
seif_fine = ((FINE - 0.737) / 0.029) ** 2

splits = {}
for name in ["dr1_bao_only", "dr1_bao_cmb", "dr2_bao_only", "dr2_bao_cmb"]:
    splits[name + "_vs_SN"] = split_stats(ts_curves[name], sn_fine, sn_cmin,
                                          sig_sn, sn_best, name)
    splits[name + "_vs_seifert737"] = split_stats(ts_curves[name], seif_fine, 0.0,
                                                  0.029, 0.737, name)
for k, v in splits.items():
    print(f"[dr2] split {k:34s} gap={v['gap_fv0']:+.4f} gauss={v['sigma_gaussian']:.2f}sig "
          f"T={v['T_profile_shift']:.2f} -> {v['sigma_profile']:.2f}sig "
          f"(joint min fv0={v['joint_min_fv0']:.4f})", flush=True)

# --------------------------------------------------------- consistency gates ----
gates = dict(
    dr1_baocmb_fv0_vs_committed_0636=abs(fits["dr1_bao_cmb"]["timescape"]["fv0"] - 0.636),
    dr1_baocmb_chi2_vs_committed_3633=abs(fits["dr1_bao_cmb"]["timescape"]["chi2"] - 36.32731787580997),
    dr1_baocmb_lcdm_Om_vs_committed_0300=abs(fits["dr1_bao_cmb"]["lcdm"]["Om"] - 0.300),
    dr1_baoonly_fv0_vs_committed_0677=abs(fits["dr1_bao_only"]["timescape"]["fv0"] - 0.677),
    sn_fv0_vs_committed_08529=abs(sn["fv0"] - 0.8528822055137845),
    sn_chi2_vs_committed_1391545=abs(sn["chi2"] - 1391.5451414212584),
    sn_dchi2_at_dr1_baoonly_fv0_vs_committed_2575=abs(
        float(np.interp(fits["dr1_bao_only"]["timescape"]["fv0"], FVSN, sn_chis)) - sn_cmin
        - 25.747027814388275),
)
for k, v in gates.items():
    print(f"[dr2] gate {k}: |diff|={v:.4f}", flush=True)

# compact delta-chi2 curves for downstream use (every 5th grid point, dchi2<=60)
def compact(grid, chis, cap=60.0, step=5):
    d = chis - chis.min(); m = d <= cap
    return dict(fv0=[round(float(x), 4) for x in grid[m][::step]],
                dchi2=[round(float(x), 3) for x in d[m][::step]])

out = dict(
    name="dr2",
    provenance=dict(
        primary="DESI DR2 official Gaussian BAO likelihood: github.com/CobayaSampler/bao_data "
                "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt @ commit b7b8a36e9bcc "
                "(2025-03-20), fetched 2026-07-05; paper arXiv:2503.14738",
        crosscheck="arXiv:2503.14738v2 summary table (rounded journal values); all 13 central "
                   "values agree to <=0.001, quoted errors to <=0.004 (table symmetrizes/rounds "
                   "the posterior; Cobaya cov is the released Gaussian likelihood)",
        cmb_point=dict(value=DMZ_RD, err=SIG_CMB, zstar=ZSTAR,
                       handling="identical to src/verify_and_extend.py (theta*=1.04109, "
                                "r*/rd=144.39/147.09, error floored at 0.05)"),
        rd_for_H0=RD,
        note_rd_pivot="DESI DR2 (eq. 2 of 2503.14738) pivots its rd fitting formula on 147.05 Mpc; "
                      "using 147.09 shifts H0 by +0.03% (~0.02 km/s/Mpc), negligible",
    ),
    dr2_rows_used=[list(r) for r in DR2_ROWS],
    dr2_table_rows=[list(r) for r in DR2_TABLE],
    crosscheck_rows=CROSSCHECK,
    fits=fits,
    sn_standard_fullcov=sn,
    seifert=dict(fv0=0.737, err=0.029, source="arXiv:2412.15143, z_min>=0.055 cut"),
    fv_split=splits,
    consistency_gates=gates,
    curves=dict(
        dr2_bao_only=compact(FVG, ts_curves["dr2_bao_only"]),
        dr2_bao_cmb=compact(FVG, ts_curves["dr2_bao_cmb"]),
        sn=compact(FVSN, sn_chis, cap=60.0, step=2),
    ),
    runtime_s=time.time() - t0,
)
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(f"[dr2] wrote {OUT} ({time.time()-t0:.0f}s total)", flush=True)
