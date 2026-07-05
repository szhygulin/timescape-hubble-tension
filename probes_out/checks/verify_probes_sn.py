# Independent adversarial spot-checks for probes P1 (fvsplit), P2 (mocks), P3 (evidence).
# Uses ONLY the verified physics primitives (D_shape_TS, D_shape_LCDM, model_vec,
# build_cov, load_sn) and re-implements every STATISTIC from scratch to confirm the
# headline numbers. Writes nothing except this script's stdout. No repo-file writes.
import os, sys, json
import numpy as np
from scipy import stats as st

WT = "/Users/s/dev/science/timescape-hubble-tension/.claude/worktrees/significance-audit"
SRC = WT + "/src"
os.chdir(SRC); sys.path.insert(0, SRC)
import fit_timescape as F
import timescape_baocmb as T
import harness as H

out = {}

# ======================================================================
# P1 fvsplit headline: S1 (SN std, full cov) x B4 (DESI DR1 BAO-only)
# parameter-shift statistic  T = min_fv[SN(fv)+BAO(fv)] - minSN - minBAO
# ======================================================================
zHD, zHEL, mb, Cf = F.load()          # 1580 SNe, full stat+sys cov
N = len(zHD)
Cinv = np.linalg.inv(Cf)
one = np.ones(N); Cinv1 = Cinv @ one; s11 = float(one @ Cinv1)

def sn_prof_chi2(fv):
    mu = 5.0*np.log10((1.0+zHEL)*F.D_shape_TS(zHD, fv))
    r = mb - mu; Cr = Cinv @ r
    return float(r @ Cr - (one @ Cr)**2/s11)

# BAO-only rows (12 DESI DR1 observables, no CMB)
rows12 = [(z,k,v,e,c) for (z,k,v,e,c) in T.BAO]
Cb = T.build_cov(rows12); Cbi = np.linalg.inv(Cb)
db = np.array([r[2] for r in rows12]); Cbd = Cbi @ db; dCd = float(db @ Cbd)
def bao_prof_chi2(fv):
    g = T.model_vec(fv, rows12)
    num = float(g @ Cbd); den = float(g @ (Cbi @ g))
    return dCd - num*num/den

grid = np.linspace(0.50, 0.96, 2301)
c_sn = np.array([sn_prof_chi2(fv) for fv in grid])
c_bao = np.array([bao_prof_chi2(fv) for fv in grid])
d_sn = c_sn - c_sn.min(); d_bao = c_bao - c_bao.min()
fv_sn_min = grid[np.argmin(c_sn)]; fv_bao_min = grid[np.argmin(c_bao)]
tot = d_sn + d_bao; j = int(np.argmin(tot))
Tstat = float(tot[j]); sigma = float(np.sqrt(max(Tstat, 0.0)))
out["P1_fvsplit_S1xB4"] = dict(
    fv_sn_min=float(fv_sn_min), fv_bao_min=float(fv_bao_min),
    fv_joint=float(grid[j]), dchi2_join=round(Tstat, 3), sigma=round(sigma, 3),
    json_says=dict(dchi2_join=19.545, sigma=4.421, fv_joint=0.712,
                   S1_fv0=0.853, B4_fv0=0.677))

# ======================================================================
# P2 mocks: (a) TS-truth dBIC tail p(Delta>=5.14); (b) fv-split 0/4000
# Independent re-implementation of the offset-profiled grid fit + mocks.
# ======================================================================
zS, zHELs, mbS, CS = H.load_sn()
assert len(zS) == N
CSinv = np.linalg.inv(CS); Lc = np.linalg.cholesky(CS)
oneS = np.ones(N); wS = CSinv @ oneS; SS = float(oneS @ wS)

FV = np.arange(0.30, 0.9901, 0.002)
OM = np.arange(0.05, 0.6001, 0.0025)
MU_TS = np.array([5.0*np.log10((1.0+zHELs)*F.D_shape_TS(zS, fv)) for fv in FV])
MU_L  = np.array([5.0*np.log10((1.0+zHELs)*F.D_shape_LCDM(zS, om)) for om in OM])

def profiled_min(MU, grid, data):
    # offset-profiled chi2 of `data` against each model row; return grid min value
    R = data[None, :] - MU
    V = CSinv @ R.T
    chi = np.einsum('kn,nk->k', R, V) - (oneS @ V)**2/SS
    return float(np.min(chi))

# real-data dBIC cross-check
DELTA_OBS = 5.138503350317478
d_TS = profiled_min(MU_TS, FV, mbS); d_L = profiled_min(MU_L, OM, mbS)
delta_real = d_TS - d_L

# TS-truth mocks (seed 2), Delta = chi2TS_min - chi2L_min
FV_TRUTH_SN = 0.8528822055137845
mu_truth = 5.0*np.log10((1.0+zHELs)*F.D_shape_TS(zS, FV_TRUTH_SN))
rng = np.random.default_rng(2)
J = 4000
noise = Lc @ rng.standard_normal((N, J))
# vectorized offset-profiled minima over grid for all mocks
def mock_min_over_grid(MU, grid, mu_truth, noise):
    Dm = mu_truth[None, :] - MU            # (K,N)
    V = CSinv @ Dm.T                       # (N,K)
    A = np.einsum('kn,nk->k', Dm, V)       # d'Ci d
    a = oneS @ V                           # 1'Ci d
    G = noise.T @ V                        # (J,K)
    b = noise.T @ wS                       # (J,)
    CH = A[None, :] + 2.0*G - (a[None, :] + b[:, None])**2/SS
    return CH.min(axis=1)
yTS = mock_min_over_grid(MU_TS, FV, mu_truth, noise)
yL  = mock_min_over_grid(MU_L, OM, mu_truth, noise)
Delta = yTS - yL
n_exc = int(np.sum(Delta >= DELTA_OBS))
out["P2_mocks_TS_truth_dBIC"] = dict(
    delta_real=round(delta_real, 4), n_exceed=n_exc, n=J,
    p_emp=n_exc/J, sigma_one_sided=round(float(st.norm.isf(n_exc/J)), 3) if n_exc>0 else None,
    json_says=dict(n_exceed=32, p_emp=0.008, sigma=2.42))

# fv-split under single-fv0 joint truth (seeds 3 SN, 4 BAO)
FV_TRUTH_JOINT = 0.6426065162907268
SPLIT_OBS = 0.853 - 0.639
mu_tj = 5.0*np.log10((1.0+zHELs)*F.D_shape_TS(zS, FV_TRUTH_JOINT))
rng3 = np.random.default_rng(3)
noise3 = Lc @ rng3.standard_normal((N, J))
# SN fv_hat per mock (argmin over FV grid, offset-profiled)
def mock_argmin(MU, grid, mu_truth, noise):
    Dm = mu_truth[None, :] - MU; V = CSinv @ Dm.T
    A = np.einsum('kn,nk->k', Dm, V); a = oneS @ V
    G = noise.T @ V; b = noise.T @ wS
    CH = A[None, :] + 2.0*G - (a[None, :] + b[:, None])**2/SS
    return grid[np.argmin(CH, axis=1)]
fv_sn_hat = mock_argmin(MU_TS, FV, mu_tj, noise3)
# BAO+CMB side (harness rows, cmb err 0.05)
rowsB = H.bao_cmb_rows(); dB = np.array([r[2] for r in rowsB])
CB = T.build_cov(rowsB); CBi = np.linalg.inv(CB); LB = np.linalg.cholesky(CB)
FVB = np.arange(0.55, 0.7301, 0.0005)
GM = np.array([T.model_vec(fv, rowsB) for fv in FVB])
P = CBi @ GM.T; den = np.einsum('kn,nk->k', GM, P)
g_t = T.model_vec(FV_TRUTH_JOINT, rowsB)
alpha_true = float((g_t @ (CBi @ dB))/(g_t @ (CBi @ g_t)))
rng4 = np.random.default_rng(4)
YB = alpha_true*g_t[:, None] + LB @ rng4.standard_normal((len(dB), J))
CHB = -(YB.T @ P)**2/den[None, :]
fv_bao_hat = FVB[np.argmin(CHB, axis=1)]
dfv = fv_sn_hat - fv_bao_hat
n_exc_split = int(np.sum(dfv >= SPLIT_OBS))
out["P2_mocks_fvsplit"] = dict(
    dfv_mean=round(float(np.mean(dfv)), 4), dfv_sd=round(float(np.std(dfv, ddof=1)), 4),
    dfv_max=round(float(np.max(dfv)), 4), split_obs=SPLIT_OBS,
    n_exceed=n_exc_split, n=J, rule_of_three_sigma=round(float(st.norm.isf(3.0/J)), 3),
    json_says=dict(n_exceed=0, dfv_sd=0.0275, dfv_max=0.1221))

# ======================================================================
# P3 evidence: standard_full lnB for prior Om~U(0.10,0.60) x fv0~U(0.50,0.95)
# Independent trapezoid quadrature of exp(-chi2/2) over each 1-param prior.
# ======================================================================
fv_e = np.linspace(0.40, 0.99, 591); om_e = np.linspace(0.05, 0.70, 651)
def sn_full_chi2_curve(grid, shape_fn):
    ch = np.empty(len(grid))
    for i, g in enumerate(grid):
        mu = 5.0*np.log10((1.0+zHEL)*shape_fn(g))
        r = mb - mu; Cr = Cinv @ r
        ch[i] = r @ Cr - (one @ Cr)**2/s11
    return ch
chiTS = sn_full_chi2_curve(fv_e, lambda fv: F.D_shape_TS(zHD, fv))
chiL  = sn_full_chi2_curve(om_e, lambda om: F.D_shape_LCDM(zHD, om))
cref = min(chiTS.min(), chiL.min())
def lnZ(grid, chi, lo, hi):
    m = (grid >= lo-1e-12) & (grid <= hi+1e-12)
    g = grid[m]; c = chi[m]
    I = np.trapz(np.exp(-0.5*(c - cref)), g) if hasattr(np, "trapz") else np.trapezoid(np.exp(-0.5*(c-cref)), g)
    return float(np.log(I) - np.log(hi - lo))
lnZ_L = lnZ(om_e, chiL, 0.10, 0.60)
lnZ_TS = lnZ(fv_e, chiTS, 0.50, 0.95)
lnB = lnZ_L - lnZ_TS
out["P3_evidence_standard_full"] = dict(
    dchi2=round(float(chiTS.min() - chiL.min()), 4),
    lnB_LCDM_over_TS=round(lnB, 4), twolnB=round(2*lnB, 4),
    json_says=dict(lnB=1.5715, twolnB=3.143, dchi2=5.1398, kass="positive favours LCDM"))

print(json.dumps(out, indent=2))
