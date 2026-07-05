#!/usr/bin/env python3
# Purpose: Frequentist calibration of the paper's model-comparison statistics by
# parametric bootstrap. (1) SN-only: distribution of Delta = chi2_TS_min - chi2_LCDM_min
# (= dBIC, equal k) under LCDM-truth and timescape-truth mocks drawn from the FULL
# Pantheon+ stat+sys covariance, converting the observed dBIC=+5.14 into p-values,
# including the paper's missing discriminating-power statement P(Delta>=5.14 | TS true).
# (2) fv0-split: under a SINGLE-fv0 timescape truth (joint best fit fv0=0.6426),
# joint SN + BAO+CMB mocks -> distribution of dfv = fv_SN - fv_BAOCMB, giving
# P(dfv >= 0.214) for the observed split (and the 0.217 variant).
# (3) Seifert-style split 0.098 (and 0.101 variant) against the same mock distribution.
# Output: probes_out/mocks.json.  Deterministic (fixed seeds).
import os, sys, json, time
import numpy as np
from scipy import stats as st

ROOT = "/Users/s/dev/science/timescape-hubble-tension/.claude/worktrees/significance-audit"
os.chdir(ROOT + "/src")
sys.path.insert(0, ROOT + "/src")

import fit_timescape as F
import timescape_baocmb as T   # module-level prints its own BAO fit banner; harmless
import harness as H

T0 = time.time()

# ------------------------------------------------------------------
# Observed values (committed provenance)
# ------------------------------------------------------------------
DELTA_OBS = 5.138503350317478          # results.json standard_full dBIC (=chi2_TS-chi2_L)
OM_TRUTH = 0.3332775919732441          # results.json standard_full Om (LCDM best fit)
FV_TRUTH_SN = 0.8528822055137845       # results.json standard_full fv0 (TS best fit)
FV_TRUTH_JOINT = 0.6426065162907268    # results_joint.json timescape fv0 (joint best fit)
SPLIT_TASK = 0.853 - 0.639             # = 0.214 (results_fvtension.json values, task spec)
SPLIT_VAE = 0.8528822055137845 - 0.636 # = 0.2169 (results.json vs results_baocmb.json)
SPLIT_SEIFERT = 0.737 - 0.639          # = 0.098 (Seifert cosmology-independent fv0)
SPLIT_SEIFERT_V = 0.737 - 0.636        # = 0.101 variant

J_SN = 4000    # mocks per SN truth
J_SPLIT = 4000 # joint mocks for the fv-split calibration

# ------------------------------------------------------------------
# SN machinery (identical conventions to fit_timescape.make_chi2)
# ------------------------------------------------------------------
zHD, zHEL, mb, C = H.load_sn()
N = len(zHD)
Cinv = np.linalg.inv(C)            # same operation as committed make_chi2
Lchol = np.linalg.cholesky(C)
one = np.ones(N)
w = Cinv @ one
S = float(one @ w)

FV_GRID = np.arange(0.30, 0.9901, 0.002)    # 346 pts, covers +-5 sigma of every truth
OM_GRID = np.arange(0.05, 0.6001, 0.0025)   # 221 pts, same range as committed scans

def mu_of_Dshape(D):
    return 5.0 * np.log10((1.0 + zHEL) * D)   # offset-free shape, as make_chi2

print(f"[{time.time()-T0:6.1f}s] building TS mu grid ({len(FV_GRID)} pts)...", flush=True)
MU_TS = np.array([mu_of_Dshape(F.D_shape_TS(zHD, fv)) for fv in FV_GRID])
print(f"[{time.time()-T0:6.1f}s] building LCDM mu grid ({len(OM_GRID)} pts)...", flush=True)
MU_L = np.array([mu_of_Dshape(F.D_shape_LCDM(zHD, om)) for om in OM_GRID])

def parab_refine(grid, y, i):
    """3-point parabola around grid minimum -> (x*, y*, railed)."""
    K = len(grid)
    if i == 0 or i == K - 1:
        return float(grid[i]), float(y[i]), True
    ym, y0, yp = y[i-1], y[i], y[i+1]
    den = ym - 2.0*y0 + yp
    if den <= 0:
        return float(grid[i]), float(y0), False
    step = grid[1] - grid[0]
    dx = 0.5 * (ym - yp) / den
    return float(grid[i] + dx*step), float(y0 - 0.125*(ym - yp)**2 / den), False

def refine_rows(grid, CH):
    """Vector parab_refine over mock rows of a (J,K) chi2 matrix."""
    idx = np.argmin(CH, axis=1)
    J = CH.shape[0]
    xs = np.empty(J); ys = np.empty(J); railed = np.zeros(J, dtype=bool)
    for j in range(J):
        xs[j], ys[j], railed[j] = parab_refine(grid, CH[j], int(idx[j]))
    return xs, ys, railed

def gen_sn_noise(J, seed):
    rng = np.random.default_rng(seed)
    return Lchol @ rng.standard_normal((N, J))   # (N,J)

def min_over_grid(MU, grid, mu_truth, Nse):
    """Offset-profiled chi2 minimum over the grid for every mock, RELATIVE to the
    per-mock constant n'Cinv n (identical for both models, cancels in Delta and is
    irrelevant to the argmin).
    chi2_rel(j,k) = d_k'Ci d_k + 2 n_j'Ci d_k - (1'Ci d_k + 1'Ci n_j)^2 / (1'Ci 1),
    with d_k = mu_truth - mu_k.  Exact algebraic expansion of make_chi2's formula."""
    Dm = mu_truth[None, :] - MU        # (K,N)
    V = Cinv @ Dm.T                    # (N,K)
    A = np.einsum('kn,nk->k', Dm, V)   # d_k'Ci d_k
    a = one @ V                        # 1'Ci d_k
    G = Nse.T @ V                      # (J,K)  n_j'Ci d_k
    b = Nse.T @ w                      # (J,)   1'Ci n_j
    CH = A[None, :] + 2.0*G - (a[None, :] + b[:, None])**2 / S
    xs, ys, railed = refine_rows(grid, CH)
    # zero-noise (deterministic) profiled distance of the truth curve to the grid family
    lam = float(np.min(A - a**2 / S))
    return xs, ys, railed, lam

def real_data_fit(MU, grid):
    """Full offset-profiled chi2 of the REAL data over the grid (cross-check)."""
    Rm = mb[None, :] - MU
    V = Cinv @ Rm.T
    chi = np.einsum('kn,nk->k', Rm, V) - (one @ V)**2 / S
    i = int(np.argmin(chi))
    x, y, railed = parab_refine(grid, chi, i)
    return x, y, railed, chi

def tail_stats(x, thr, side, label):
    """Empirical + Gaussian + GPD(peaks-over-threshold) tail probability that x is
    beyond thr on the given side. sigma equivalents are ONE-SIDED: sigma=Phi^-1(1-p)."""
    x = np.asarray(x, dtype=float); n = len(x)
    y = x if side == "ge" else -x
    t = thr if side == "ge" else -thr
    cnt = int(np.sum(y >= t))
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    z = (t - np.mean(y)) / np.std(y, ddof=1)
    p_gauss = float(st.norm.sf(z))
    out = dict(label=label, threshold=float(thr), side=side, n_mocks=n,
               n_exceed=cnt, p_emp=cnt / n, mock_mean=m, mock_sd=s,
               z_gauss=float(z), p_gauss=p_gauss,
               sigma_gauss_one_sided=(float(st.norm.isf(p_gauss)) if p_gauss > 0 else None))
    u = float(np.quantile(y, 0.95))
    exc = y[y > u] - u
    if t > u and len(exc) >= 50:
        c, loc, sc = st.genpareto.fit(exc, floc=0.0)
        p_gpd = float(len(exc) / n * st.genpareto.sf(t - u, c, loc=0.0, scale=sc))
        out["gpd"] = dict(u=u, n_exc=int(len(exc)), shape=float(c), scale=float(sc),
                          p_gpd=p_gpd,
                          sigma_gpd_one_sided=(float(st.norm.isf(p_gpd)) if p_gpd > 0 else None),
                          note="POT fit to top 5% of mocks; extrapolation beyond mock max is model-dependent")
    if cnt == 0:
        out["hard_bound"] = dict(p_lt=1.0 / n, rule_of_three_p95_upper=3.0 / n,
                                 note="zero of n mocks exceeded; p < 1/n at face value, 95% CL upper bound 3/n")
    return out

QS = [0.001, 0.01, 0.025, 0.05, 0.16, 0.25, 0.50, 0.75, 0.84, 0.95, 0.975, 0.99, 0.999]
def quantiles(x):
    return {str(q): float(v) for q, v in zip(QS, np.quantile(x, QS))}

results = dict(script="src/probes/mocks.py", seeds=dict(lcdm_truth=1, ts_truth=2, split_sn=3, split_bao=4),
               N_sn=N, J_sn=J_SN, J_split=J_SPLIT,
               grids=dict(fv=[float(FV_GRID[0]), float(FV_GRID[-1]), len(FV_GRID)],
                          om=[float(OM_GRID[0]), float(OM_GRID[-1]), len(OM_GRID)]))

# ------------------------------------------------------------------
# Cross-check: real-data fits on our grids must reproduce the paper
# ------------------------------------------------------------------
fv_obs, chiTS_obs, _, _ = real_data_fit(MU_TS, FV_GRID)
om_obs, chiL_obs, _, _ = real_data_fit(MU_L, OM_GRID)
results["real_data_check"] = dict(
    fv0_SN=fv_obs, Om=om_obs, chi2_TS=chiTS_obs, chi2_L=chiL_obs,
    Delta_mygrid=chiTS_obs - chiL_obs, Delta_committed=DELTA_OBS,
    committed=dict(fv0=FV_TRUTH_SN, Om=OM_TRUTH, chi2_TS=1391.5451414212584, chi2_L=1386.406638070941))
print(f"[{time.time()-T0:6.1f}s] real-data check: fv0={fv_obs:.4f} Om={om_obs:.4f} "
      f"Delta={chiTS_obs-chiL_obs:+.3f} (committed {DELTA_OBS:+.3f})", flush=True)

# ------------------------------------------------------------------
# Part 1a: LCDM-truth mocks
# ------------------------------------------------------------------
mu_truth_L = mu_of_Dshape(F.D_shape_LCDM(zHD, OM_TRUTH))
print(f"[{time.time()-T0:6.1f}s] part 1a: {J_SN} LCDM-truth mocks...", flush=True)
Nse = gen_sn_noise(J_SN, 1)
omL_hat, yL_L, rail_a, lam_L_vs_L = min_over_grid(MU_L, OM_GRID, mu_truth_L, Nse)
fvL_hat, yL_T, rail_b, lam_L_vs_TS = min_over_grid(MU_TS, FV_GRID, mu_truth_L, Nse)
del Nse
Delta_lcdm = yL_T - yL_L
results["part1_lcdm_truth"] = dict(
    truth=dict(model="LCDM", Om=OM_TRUTH),
    noncentrality_min_profiled_dist_truth_to_TS_family=lam_L_vs_TS,
    Delta_quantiles=quantiles(Delta_lcdm),
    Delta_mean=float(np.mean(Delta_lcdm)), Delta_sd=float(np.std(Delta_lcdm, ddof=1)),
    frac_Delta_negative=float(np.mean(Delta_lcdm < 0)),
    railed_frac=dict(om=float(np.mean(rail_a)), fv=float(np.mean(rail_b))),
    fv_hat=dict(mean=float(np.mean(fvL_hat)), sd=float(np.std(fvL_hat, ddof=1))),
    om_hat=dict(mean=float(np.mean(omL_hat)), sd=float(np.std(omL_hat, ddof=1))),
    p_le_obs=tail_stats(Delta_lcdm, DELTA_OBS, "le", "P(Delta<=+5.14 | LCDM true): is the observed LCDM win unusually WEAK?"),
    p_ge_obs=tail_stats(Delta_lcdm, DELTA_OBS, "ge", "P(Delta>=+5.14 | LCDM true): is the observed LCDM win unusually STRONG?"),
    tail_note=("Under LCDM truth Delta is typically positive; the lower tail P(Delta<=obs) tests "
               "whether the observed win is suspiciously small (TS mimicking), the upper tail whether "
               "it is larger than LCDM-truth sampling scatter explains. Neither is the evidence-against-TS "
               "statement; that is part1_ts_truth.p_ge_obs."))
print(f"[{time.time()-T0:6.1f}s] LCDM-truth: Delta mean={np.mean(Delta_lcdm):+.2f} "
      f"sd={np.std(Delta_lcdm):.2f} P(<=obs)={np.mean(Delta_lcdm<=DELTA_OBS):.4f}", flush=True)

# ------------------------------------------------------------------
# Part 1b: timescape-truth mocks (SN best fit fv0=0.8529)
# ------------------------------------------------------------------
mu_truth_TS = mu_of_Dshape(F.D_shape_TS(zHD, FV_TRUTH_SN))
print(f"[{time.time()-T0:6.1f}s] part 1b: {J_SN} TS-truth mocks...", flush=True)
Nse = gen_sn_noise(J_SN, 2)
omT_hat, yT_L, rail_c, lam_TS_vs_L = min_over_grid(MU_L, OM_GRID, mu_truth_TS, Nse)
fvT_hat, yT_T, rail_d, _ = min_over_grid(MU_TS, FV_GRID, mu_truth_TS, Nse)
del Nse
Delta_ts = yT_T - yT_L
results["part1_ts_truth"] = dict(
    truth=dict(model="timescape", fv0=FV_TRUTH_SN),
    noncentrality_min_profiled_dist_truth_to_LCDM_family=lam_TS_vs_L,
    Delta_quantiles=quantiles(Delta_ts),
    Delta_mean=float(np.mean(Delta_ts)), Delta_sd=float(np.std(Delta_ts, ddof=1)),
    frac_Delta_positive=float(np.mean(Delta_ts > 0)),
    railed_frac=dict(om=float(np.mean(rail_c)), fv=float(np.mean(rail_d))),
    fv_hat=dict(mean=float(np.mean(fvT_hat)), sd=float(np.std(fvT_hat, ddof=1))),
    p_ge_obs=tail_stats(Delta_ts, DELTA_OBS, "ge",
                        "P(Delta>=+5.14 | TS true): probability of the observed LCDM win if timescape were right (discriminating power / evidence against TS)"),
    tail_note=("One-sided upper tail: timescape is the null; large positive Delta is evidence against it. "
               "This is the frequentist p-value that calibrates the observed dBIC=+5.14."))
print(f"[{time.time()-T0:6.1f}s] TS-truth: Delta mean={np.mean(Delta_ts):+.2f} "
      f"sd={np.std(Delta_ts):.2f} P(>=obs)={np.mean(Delta_ts>=DELTA_OBS):.4f}", flush=True)

# ------------------------------------------------------------------
# Part 2: fv-split calibration under single-fv0 timescape truth (joint best fit)
# ------------------------------------------------------------------
print(f"[{time.time()-T0:6.1f}s] part 2: {J_SPLIT} joint mocks at fv0={FV_TRUTH_JOINT:.4f}...", flush=True)
mu_truth_J = mu_of_Dshape(F.D_shape_TS(zHD, FV_TRUTH_JOINT))
Nse = gen_sn_noise(J_SPLIT, 3)
fv_sn_hat, _, rail_sn, _ = min_over_grid(MU_TS, FV_GRID, mu_truth_J, Nse)
del Nse

# BAO+CMB side: harness ROWS (DESI DR1 + Planck point with the corrected 0.05 error)
rowsB = H.bao_cmb_rows()
dB = np.array([r[2] for r in rowsB])
CB = T.build_cov(rowsB)
CBinv = np.linalg.inv(CB)
LB = np.linalg.cholesky(CB)
FVB_GRID = np.arange(0.55, 0.7301, 0.0005)
GM = np.array([T.model_vec(fv, rowsB) for fv in FVB_GRID])   # (K2,13)
g_t = T.model_vec(FV_TRUTH_JOINT, rowsB)
alpha_true = float((g_t @ (CBinv @ dB)) / (g_t @ (CBinv @ g_t)))
P = CBinv @ GM.T                            # (13,K2)
den = np.einsum('kn,nk->k', GM, P)          # g_k'Ci g_k

# observed BAO+CMB fit on the real data (cross-check + refined minimum + dchi2=1 width)
q_obs = (dB @ P)**2 / den
chiB_obs_curve = float(dB @ (CBinv @ dB)) - q_obs
iB = int(np.argmin(chiB_obs_curve))
fvB_obs, chiB_obs, _ = parab_refine(FVB_GRID, chiB_obs_curve, iB)
dchi = chiB_obs_curve - chiB_obs
lo = float(np.interp(1.0, dchi[:iB+1][::-1], FVB_GRID[:iB+1][::-1]))
hi = float(np.interp(1.0, dchi[iB:], FVB_GRID[iB:]))
results["baocmb_real_data_check"] = dict(
    fv0_BAOCMB_refined=fvB_obs, chi2=chiB_obs, dof=len(rowsB)-2,
    fv0_err_dchi2eq1=[fvB_obs - lo, hi - fvB_obs],
    alpha_at_joint_truth=alpha_true,
    committed=dict(fv0=0.636, chi2=36.32731787580997, source="results_baocmb.json (grid step 0.001)"))
print(f"[{time.time()-T0:6.1f}s] BAO+CMB real data: fv0={fvB_obs:.4f} "
      f"(+{hi-fvB_obs:.4f}/-{fvB_obs-lo:.4f}) chi2={chiB_obs:.2f}", flush=True)

rngB = np.random.default_rng(4)
YB = alpha_true * g_t[:, None] + LB @ rngB.standard_normal((len(dB), J_SPLIT))  # (13,J)
CHB = -(YB.T @ P)**2 / den[None, :]         # chi2 up to per-mock const
fv_bao_hat, _, rail_bao = refine_rows(FVB_GRID, CHB)

dfv = fv_sn_hat - fv_bao_hat
results["part2_fv_split"] = dict(
    truth=dict(model="timescape single fv0", fv0=FV_TRUTH_JOINT, alpha_baocmb=alpha_true,
               bao_rows=len(rowsB), cmb_err=float(rowsB[-1][3])),
    fv_sn_hat=dict(mean=float(np.mean(fv_sn_hat)), sd=float(np.std(fv_sn_hat, ddof=1)),
                   bias=float(np.mean(fv_sn_hat) - FV_TRUTH_JOINT), railed_frac=float(np.mean(rail_sn))),
    fv_bao_hat=dict(mean=float(np.mean(fv_bao_hat)), sd=float(np.std(fv_bao_hat, ddof=1)),
                    bias=float(np.mean(fv_bao_hat) - FV_TRUTH_JOINT), railed_frac=float(np.mean(rail_bao))),
    dfv_quantiles=quantiles(dfv), dfv_mean=float(np.mean(dfv)), dfv_sd=float(np.std(dfv, ddof=1)),
    dfv_max=float(np.max(dfv)), dfv_min=float(np.min(dfv)),
    observed_split_task=tail_stats(dfv, SPLIT_TASK, "ge", "P(dfv>=0.214 | single fv0): task-specified split 0.853-0.639"),
    observed_split_variant=tail_stats(dfv, SPLIT_VAE, "ge", "P(dfv>=0.2169): results.json 0.8529 vs results_baocmb.json 0.636"))
print(f"[{time.time()-T0:6.1f}s] split: dfv mean={np.mean(dfv):+.4f} sd={np.std(dfv):.4f} "
      f"max={np.max(dfv):+.4f} P(>=0.214)={np.mean(dfv>=SPLIT_TASK):.5f}", flush=True)

# ------------------------------------------------------------------
# Part 3: Seifert-style split against the same mock distribution
# ------------------------------------------------------------------
results["part3_seifert_split"] = dict(
    observed_split=tail_stats(dfv, SPLIT_SEIFERT, "ge", "P(dfv>=0.098): Seifert fv0=0.737 vs BAO+CMB 0.639"),
    observed_split_variant=tail_stats(dfv, SPLIT_SEIFERT_V, "ge", "P(dfv>=0.101): Seifert 0.737 vs verify_and_extend 0.636"),
    caveat=("Seifert et al.'s fv0=0.737 comes from their own Tripp standardisation + covariance, "
            "which this repo cannot reproduce; the SN-side sampling error used here is the mock error "
            "of THIS pipeline's m_b_corr full-covariance fit (sd of fv_sn_hat above), a proxy that is "
            "likely narrower than Seifert's actual posterior width."))

results["runtime_s"] = round(time.time() - T0, 1)
out_path = os.path.join(ROOT, "probes_out", "mocks.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"[{time.time()-T0:6.1f}s] wrote {out_path}", flush=True)
