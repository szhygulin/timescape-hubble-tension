# Purpose: recompute the SN-vs-BAO(+CMB) void-fraction (fv0) tension PROPERLY from
# full profile-likelihood Delta-chi2(fv0) curves via the parameter-shift statistic
# (Delta-chi2_join = min_fv[SN+BAO] - (min SN + min BAO), 1 dof), under an explicit
# systematics ladder: (i) committed pipeline errors; (ii) CMB acoustic-point error
# floored at 0.05 vs propagated 0.0272; (iii) Nazer & Wiltshire 13% fv0 systematic
# grafted onto the BAO+CMB profile (min-plus convolution); (iv) BAO-only (no CMB).
# Replaces the paper's "~1-2 sigma" Gaussian propagation with a defensible number
# per (SN reduction) x (BAO-side treatment).
import os, sys, json, time
import numpy as np

WT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = WT + "/src"
OUTJ = WT + "/probes_out/fvsplit.json"
OUTN = WT + "/probes_out/fvsplit_curves.npz"
os.chdir(SRC)
sys.path.insert(0, SRC)

import fit_timescape as F           # SN machinery (verbatim Dam+2017 distances)
import timescape_baocmb as T        # BAO rows / model_vec / build_cov (prints its own banner on import)
import fit_tripp as FT              # profiled-Tripp -2lnL machinery

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ----------------------------------------------------------------------
# 1. Load Pantheon+ once (all columns needed by every SN reduction)
# ----------------------------------------------------------------------
def load_all():
    with open(F.DATA) as f:
        header = f.readline().split(); idx = {n: i for i, n in enumerate(header)}
        rows = [ln.split() for ln in f]
    col = lambda n: np.array([float(r[idx[n]]) for r in rows])
    names = ["zHD","zHEL","m_b_corr","mB","x1","c","mBERR","x1ERR","cERR","x0",
             "COV_x1_c","COV_x1_x0","COV_c_x0","m_b_corr_err_VPEC"]
    d = {n: col(n) for n in names}
    iscal = np.array([int(float(r[idx["IS_CALIBRATOR"]])) for r in rows])
    m = (iscal == 0) & (d["zHD"] > 0.01)
    with open(F.COV) as f:
        n = int(f.readline())
    Cfull = np.loadtxt(F.COV, skiprows=1).reshape(n, n)[np.ix_(m, m)]
    return {k: v[m] for k, v in d.items()}, Cfull

log("loading Pantheon+ data + 1580x1580 stat+sys covariance ...")
D, Cfull = load_all()
N = len(D["zHD"])
zHD, zHEL, mb = D["zHD"], D["zHEL"], D["m_b_corr"]
A0, B0 = 0.131, 2.663                       # frozen (alpha,beta) as in fit_robust.py / fig_decomp.py
mag_tripp = D["mB"] + A0*D["x1"] - B0*D["c"]
Cinv = np.linalg.inv(Cfull)
one = np.ones(N); Cinv1 = Cinv @ one; s11 = one @ Cinv1
w_diag = 1.0/np.diag(Cfull); sw = w_diag.sum()
log(f"N = {N} SNe; covariance inverted")

sub075 = D["zHD"] > 0.075
D075 = {k: v[sub075] for k, v in D.items()}
n2_full_factory_cache = {}

# ----------------------------------------------------------------------
# 2. SN profile curves on a common dense fv0 grid
#    S1 standard m_b_corr | full stat+sys cov      (paper's headline reduction)
#    S2 raw Tripp (frozen a,b) | diagonal cov      (fig_decomp pro-timescape corner)
#    S3 profiled Tripp (a,b,sig_int) | stat-only, z>0.01   (results.json tripp_full)
#    S4 profiled Tripp, z>0.075                    (most BAO-friendly SN reduction)
# ----------------------------------------------------------------------
fv_sn = np.linspace(0.500, 0.960, 461)      # step 0.001; extends past 0.90 to avoid grid-clipping
c_s1 = np.empty_like(fv_sn); c_s2 = np.empty_like(fv_sn)
c_s3 = np.empty_like(fv_sn); c_s4 = np.empty_like(fv_sn)
log(f"scanning {len(fv_sn)} fv0 points x 4 SN reductions (S3/S4 refit alpha,beta,sig_int per point) ...")
for i, fv in enumerate(fv_sn):
    shape = F.D_shape_TS(zHD, fv)
    mu = 5.0*np.log10((1.0+zHEL)*shape)
    # S1: full covariance, m_b_corr, analytic offset marginalisation (== F.make_chi2)
    r = mb - mu; Cr = Cinv @ r
    c_s1[i] = r @ Cr - (one @ Cr)**2/s11
    # S2: diagonal covariance, frozen-Tripp magnitude
    r2 = mag_tripp - mu
    c_s2[i] = np.sum(w_diag*r2*r2) - (np.sum(w_diag*r2))**2/sw
    # S3: profiled Tripp, full sample (same mask as S1)
    c_s3[i] = FT.best(D, mu)[0]
    # S4: profiled Tripp, z>0.075
    c_s4[i] = FT.best(D075, mu[sub075])[0]
    if i % 50 == 0: log(f"  fv0={fv:.3f} ({i+1}/{len(fv_sn)})")
log("SN scans done")

# ----------------------------------------------------------------------
# 3. BAO-side profile curves (alpha = c/(Hbar0 rd) profiled analytically)
#    B1 BAO+CMB, cmb_err=0.05 floor (committed pipeline: verify_and_extend.py)
#    B2 BAO+CMB, cmb_err=propagated theta* error (no floor)
#    B4 BAO only (12 DESI DR1 points, no CMB, no r_d amplitude assumption)
# ----------------------------------------------------------------------
theta100, sig100 = 1.04109, 0.00030
DM_star = (100.0/theta100)*(144.39/147.09)      # 94.290 (committed column pairing kept verbatim)
sig_prop = DM_star*(sig100/theta100)            # 0.02717 propagated; committed code floors at 0.05
ZSTAR = 1089.80
rows13_e050 = [(z,k,v,e,c) for (z,k,v,e,c) in T.BAO] + [(ZSTAR,"DM",DM_star,0.05,None)]
rows13_e027 = [(z,k,v,e,c) for (z,k,v,e,c) in T.BAO] + [(ZSTAR,"DM",DM_star,sig_prop,None)]
rows12      = [(z,k,v,e,c) for (z,k,v,e,c) in T.BAO]

fv_bao = np.linspace(0.300, 0.995, 1391)        # step 0.0005
log(f"building timescape model vectors for {len(fv_bao)} fv0 points x 13 BAO+CMB observables ...")
G13 = np.array([T.model_vec(fv, rows13_e050) for fv in fv_bao])   # model_vec ignores value/err fields
G12 = G13[:, :12]

def prof_chi2(G, rows):
    C = T.build_cov(rows); Ci = np.linalg.inv(C)
    d = np.array([r[2] for r in rows])
    Cd = Ci @ d; dCd = float(d @ Cd)
    num = G @ Cd
    den = np.einsum('ij,jk,ik->i', G, Ci, G)
    return dCd - num**2/den

c_b1 = prof_chi2(G13, rows13_e050)
c_b2 = prof_chi2(G13, rows13_e027)
c_b4 = prof_chi2(G12, rows12)
log("BAO profiles done")

# ----------------------------------------------------------------------
# 4. Delta-chi2 curves, minima, Delta-chi2=1 intervals
# ----------------------------------------------------------------------
def summarize(grid, chi):
    i = int(np.argmin(chi)); dchi = chi - chi[i]
    lo = np.interp(1.0, dchi[:i+1][::-1], grid[:i+1][::-1]) if (i > 0 and dchi[0] >= 1.0) else None
    hi = np.interp(1.0, dchi[i:], grid[i:]) if (i < len(grid)-1 and dchi[-1] >= 1.0) else None
    railed = (i == 0) or (i == len(grid)-1) or lo is None or hi is None
    return dict(fv0=float(grid[i]), chi2_min=float(chi[i]),
                err_lo=(float(grid[i]-lo) if lo is not None else None),
                err_hi=(float(hi-grid[i]) if hi is not None else None),
                railed=bool(railed)), dchi

S = {}
S["S1_standard_fullcov"], d_s1 = summarize(fv_sn, c_s1)
S["S2_trippfrozen_diag"], d_s2 = summarize(fv_sn, c_s2)
S["S3_tripp_prof_z001"],  d_s3 = summarize(fv_sn, c_s3)
S["S4_tripp_prof_z075"],  d_s4 = summarize(fv_sn, c_s4)
B = {}
B["B1_baocmb_e050"], d_b1 = summarize(fv_bao, c_b1)
B["B2_baocmb_e027"], d_b2 = summarize(fv_bao, c_b2)
B["B4_baoonly"],     d_b4 = summarize(fv_bao, c_b4)
for k, v in {**S, **B}.items(): log(f"  {k}: fv0={v['fv0']:.4f} +{v['err_hi']}/-{v['err_lo']} chi2min={v['chi2_min']:.2f} railed={v['railed']}")

# Seifert cosmology-independent anchor: fv0=0.737. Two uncertainty variants:
#  - assumed +/-0.02 (original placeholder, kept for comparison), and
#  - the PUBLISHED +/-0.029 from Seifert et al. (2025, MNRAS Lett. 537, L55;
#    arXiv:2412.15143), the same +/-0.029 the tex already quotes at Sec. BAO+CMB.
# Analytic Gaussian Delta-chi2 in each case.
SEIF_FV, SEIF_ERR = 0.737, 0.020
SEIF_ERR_PUB = 0.029
S["S5_seifert_anchor"] = dict(fv0=SEIF_FV, chi2_min=0.0, err_lo=SEIF_ERR, err_hi=SEIF_ERR,
                              railed=False, note="uncertainty ASSUMED +/-0.02 (placeholder)")
S["S5_seifert_anchor_e029"] = dict(fv0=SEIF_FV, chi2_min=0.0, err_lo=SEIF_ERR_PUB, err_hi=SEIF_ERR_PUB,
                                   railed=False, note="PUBLISHED +/-0.029 (Seifert et al. 2025, arXiv:2412.15143)")

# ----------------------------------------------------------------------
# 5. Nazer & Wiltshire 13% fv0 systematic grafted onto BAO+CMB (B3):
#    min-plus (infimal) convolution of the B1 profile with a Gaussian penalty
#    ((fv-fv')/sigma_sys)^2, sigma_sys = 0.13 * fv0(B1).
# ----------------------------------------------------------------------
sigma_sys = 0.13*B["B1_baocmb_e050"]["fv0"]
fine = np.linspace(0.500, 0.960, 2301)          # common grid for joint minimisation, step 0.0002
pen = ((fine[:, None] - fv_bao[None, :])/sigma_sys)**2
d_b3_fine = (d_b1[None, :] + pen).min(axis=1)
d_b3_fine -= d_b3_fine.min()
# summarize B3 on the fine grid
i3 = int(np.argmin(d_b3_fine))
lo3 = np.interp(1.0, d_b3_fine[:i3+1][::-1], fine[:i3+1][::-1])
hi3 = np.interp(1.0, d_b3_fine[i3:], fine[i3:])
B["B3_baocmb_e050_nazersys"] = dict(fv0=float(fine[i3]), chi2_min=None,
                                    err_lo=float(fine[i3]-lo3), err_hi=float(hi3-fine[i3]),
                                    railed=False, sigma_sys=float(sigma_sys),
                                    note="B1 profile min-plus-convolved with Gaussian width 0.13*fv0_B1")
log(f"  B3 (B1 + 13% sys conv): fv0={fine[i3]:.4f} +/-{(hi3-lo3)/2:.4f} (sigma_sys={sigma_sys:.4f})")

# ----------------------------------------------------------------------
# 6. Parameter-shift tension per (SN curve) x (BAO curve)
# ----------------------------------------------------------------------
def dcurve_on_fine(grid, dchi):
    return np.interp(fine, grid, dchi)

sn_fine = {
    "S1_standard_fullcov": dcurve_on_fine(fv_sn, d_s1),
    "S2_trippfrozen_diag": dcurve_on_fine(fv_sn, d_s2),
    "S3_tripp_prof_z001":  dcurve_on_fine(fv_sn, d_s3),
    "S4_tripp_prof_z075":  dcurve_on_fine(fv_sn, d_s4),
    "S5_seifert_anchor":   ((fine - SEIF_FV)/SEIF_ERR)**2,
    "S5_seifert_anchor_e029": ((fine - SEIF_FV)/SEIF_ERR_PUB)**2,
}
bao_fine = {
    "B1_baocmb_e050":          dcurve_on_fine(fv_bao, d_b1),
    "B2_baocmb_e027":          dcurve_on_fine(fv_bao, d_b2),
    "B3_baocmb_e050_nazersys": d_b3_fine,
    "B4_baoonly":              dcurve_on_fine(fv_bao, d_b4),
}

def sym_sigma(entry):
    lo = entry["err_lo"] if entry["err_lo"] is not None else entry["err_hi"]
    hi = entry["err_hi"] if entry["err_hi"] is not None else entry["err_lo"]
    return 0.5*(lo + hi)

matrix = {}
for sname, sd in sn_fine.items():
    row = {}
    s_entry = S[sname]
    for bname, bd in bao_fine.items():
        b_entry = B[bname]
        tot = sd + bd
        j = int(np.argmin(tot))
        dchi2 = float(tot[j])
        sigma = float(np.sqrt(max(dchi2, 0.0)))
        # Gaussian cross-check (symmetrized 1-sigma widths)
        gap = abs(s_entry["fv0"] - b_entry["fv0"])
        gsig = gap/np.sqrt(sym_sigma(s_entry)**2 + sym_sigma(b_entry)**2)
        row[bname] = dict(dchi2_join=round(dchi2, 3), sigma=round(sigma, 3),
                          fv_joint=float(fine[j]),
                          sn_dchi2_at_joint=round(float(sd[j]), 3),
                          bao_dchi2_at_joint=round(float(bd[j]), 3),
                          gap_fv0=round(gap, 4), gaussian_sigma=round(float(gsig), 3),
                          joint_at_edge=bool(j in (0, len(fine)-1)))
    matrix[sname] = row
log("tension matrix done")

# ----------------------------------------------------------------------
# 7. Cross-checks vs committed numbers (results_baocmb_dr1.json core block)
# ----------------------------------------------------------------------
crosschecks = dict(
    SN_S1_dchi2_at_fv0677=float(np.interp(0.677, fv_sn, d_s1)),        # committed 25.747
    BAOonly_dchi2_at_fv0853=float(np.interp(0.853, fv_bao, d_b4)),     # committed 80.733
    committed_SN_dchi2_at_BAOfv=25.747, committed_BAO_dchi2_at_SNfv=80.733,
    B1_fv0_err_regenerated=dict(fv0=B["B1_baocmb_e050"]["fv0"],
                                err_lo=B["B1_baocmb_e050"]["err_lo"],
                                err_hi=B["B1_baocmb_e050"]["err_hi"],
                                paper_quote="0.636 +/- 0.007 (tex:299, no committed provenance)"),
    S1_fv0_unclipped=dict(fv0=S["S1_standard_fullcov"]["fv0"],
                          err_lo=S["S1_standard_fullcov"]["err_lo"],
                          err_hi=S["S1_standard_fullcov"]["err_hi"],
                          note="grid extends to 0.96, so upper error is NOT clipped at 0.900 "
                               "(results.json quotes +0.0471, clipped)"),
)

# ----------------------------------------------------------------------
# 8. Systematics ladder summary (BAO side), floor cost, paper 1-2sigma repro
# ----------------------------------------------------------------------
ladder = dict(
    i_committed_pipeline=dict(desc="published DESI errors + CMB point err floored at 0.05 (committed)",
                              sigma_vs_S1=matrix["S1_standard_fullcov"]["B1_baocmb_e050"]["sigma"]),
    ii_floor_cost=dict(desc="cmb_err 0.05 floor vs propagated theta* error (no floor)",
                       sig_cmb_propagated=float(sig_prop),
                       B1_fv0_width=sym_sigma(B["B1_baocmb_e050"]),
                       B2_fv0_width=sym_sigma(B["B2_baocmb_e027"]),
                       sigma_vs_S1_e050=matrix["S1_standard_fullcov"]["B1_baocmb_e050"]["sigma"],
                       sigma_vs_S1_e027=matrix["S1_standard_fullcov"]["B2_baocmb_e027"]["sigma"]),
    iii_nazer_sys=dict(desc="Nazer&Wiltshire 13% fv0 systematic as Gaussian width on BAO+CMB profile",
                       sigma_sys=float(sigma_sys),
                       sigma_vs_S1=matrix["S1_standard_fullcov"]["B3_baocmb_e050_nazersys"]["sigma"],
                       sigma_vs_S3=matrix["S3_tripp_prof_z001"]["B3_baocmb_e050_nazersys"]["sigma"],
                       sigma_vs_S4=matrix["S4_tripp_prof_z075"]["B3_baocmb_e050_nazersys"]["sigma"],
                       sigma_vs_S5=matrix["S5_seifert_anchor"]["B3_baocmb_e050_nazersys"]["sigma"],
                       sigma_vs_S5_e020=matrix["S5_seifert_anchor"]["B3_baocmb_e050_nazersys"]["sigma"],
                       sigma_vs_S5_e029=matrix["S5_seifert_anchor_e029"]["B3_baocmb_e050_nazersys"]["sigma"],
                       seifert_note="S5 = Seifert cosmology-independent anchor fv0=0.737; e020 assumed, "
                                    "e029 = published Seifert et al. (2025) uncertainty. These are the "
                                    "'anchor-substituted softening' figures quoted in Sec. BAO+CMB."),
    iv_bao_only=dict(desc="BAO only: no CMB point, no r_d amplitude assumption, no CMB systematics",
                     sigma_vs_S1=matrix["S1_standard_fullcov"]["B4_baoonly"]["sigma"],
                     sigma_vs_S4_floor=matrix["S4_tripp_prof_z075"]["B4_baoonly"]["sigma"]),
)

fvB1 = B["B1_baocmb_e050"]["fv0"]
paper_repro = dict(
    desc="what assumption set reproduces the paper's '~1-2 sigma' (tex:311-312, abstract tex:71-73)",
    anchor0737_sysonly=round(abs(SEIF_FV - fvB1)/sigma_sys, 3),
    anchor0737_quadrature=round(abs(SEIF_FV - fvB1)/np.sqrt(sigma_sys**2 + SEIF_ERR**2
                                    + sym_sigma(B["B1_baocmb_e050"])**2), 3),
    tripp_z075_quadrature=round(abs(S["S4_tripp_prof_z075"]["fv0"] - fvB1)
                                /np.sqrt(sigma_sys**2 + sym_sigma(S["S4_tripp_prof_z075"])**2
                                         + sym_sigma(B["B1_baocmb_e050"])**2), 3),
    tripp_full_quadrature=round(abs(S["S3_tripp_prof_z001"]["fv0"] - fvB1)
                                /np.sqrt(sigma_sys**2 + sym_sigma(S["S3_tripp_prof_z001"])**2
                                         + sym_sigma(B["B1_baocmb_e050"])**2), 3),
    standard_full_quadrature=round(abs(S["S1_standard_fullcov"]["fv0"] - fvB1)
                                   /np.sqrt(sigma_sys**2 + sym_sigma(S["S1_standard_fullcov"])**2
                                            + sym_sigma(B["B1_baocmb_e050"])**2), 3),
    statement="'1-2 sigma' requires BOTH (a) replacing the paper's own SN fit by the "
              "cosmology-independent anchor 0.737 or a Tripp reduction, AND (b) grafting the "
              "13% CMB-power-spectrum systematic (Nazer&Wiltshire 2015, recombination physics "
              "on a CMB-spectrum-derived fv0) onto the geometric BAO-shape fit, which the paper "
              "itself shows is r_d/amplitude-independent. The paper's own standard full-cov SN "
              "fit stays above 2 sigma even then.",
)

# ----------------------------------------------------------------------
# 9. Persist
# ----------------------------------------------------------------------
np.savez(OUTN, fv_sn=fv_sn, dchi2_S1=d_s1, dchi2_S2=d_s2, dchi2_S3=d_s3, dchi2_S4=d_s4,
         fv_bao=fv_bao, dchi2_B1=d_b1, dchi2_B2=d_b2, dchi2_B4=d_b4,
         fine=fine, dchi2_B3_fine=d_b3_fine,
         chi2_S1=c_s1, chi2_S2=c_s2, chi2_S3=c_s3, chi2_S4=c_s4,
         chi2_B1=c_b1, chi2_B2=c_b2, chi2_B4=c_b4)

headline = dict(
    pairing="S1_standard_fullcov x B4_baoonly",
    dchi2_join=matrix["S1_standard_fullcov"]["B4_baoonly"]["dchi2_join"],
    sigma=matrix["S1_standard_fullcov"]["B4_baoonly"]["sigma"],
    justification="SN side = the paper's own headline reduction (m_b_corr, FULL stat+sys "
                  "covariance, the same one that gives dBIC=+5.1 for LCDM); BAO side = DESI DR1 "
                  "BAO alone, which needs NO CMB point, NO r_d calibration and NO CMB systematic "
                  "-- every rung of the systematics ladder is bypassed by construction, so this "
                  "number cannot be softened by the Nazer 13% argument.",
    floor=dict(pairing="S4_tripp_prof_z075 x B4_baoonly",
               sigma=matrix["S4_tripp_prof_z075"]["B4_baoonly"]["sigma"],
               note="most BAO-friendly SN reduction vs cleanest BAO side = absolute floor"),
)

out = dict(
    name="fvsplit",
    provenance=dict(script="src/probes/fvsplit.py",
                    sn_grid=[0.500, 0.960, 461], bao_grid=[0.300, 0.995, 1391],
                    fine_grid=[0.500, 0.960, 2301],
                    cmb_point=float(DM_star), cmb_err_floor=0.05, cmb_err_propagated=float(sig_prop),
                    sigma_sys_nazer=float(sigma_sys), seifert_anchor=[SEIF_FV, SEIF_ERR],
                    seifert_anchor_published=[SEIF_FV, SEIF_ERR_PUB],
                    frozen_tripp=[A0, B0], N_sn=N,
                    stat="parameter-shift: dchi2_join = min_fv[SN(fv)+BAO(fv)] - (min SN + min BAO), "
                         "1 dof, sigma = sqrt(dchi2_join)"),
    sn_curves=S, bao_curves=B,
    tension_matrix=matrix,
    ladder=ladder,
    paper_1to2sigma_reproduction=paper_repro,
    crosschecks_vs_committed=crosschecks,
    headline=headline,
    runtime_s=round(time.time()-t0, 1),
)
with open(OUTJ, "w") as f:
    json.dump(out, f, indent=2)
log(f"wrote {OUTJ} and {OUTN}")
print(json.dumps(dict(headline=headline, ladder=ladder, paper_repro=paper_repro), indent=2))
