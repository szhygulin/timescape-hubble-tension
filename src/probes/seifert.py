# Probe "seifert": reproduce the Seifert et al. (2025, MNRAS Lett. 537, L55; arXiv:2412.15143)
# COSMOLOGY-INDEPENDENT Pantheon+ covariance analysis and run OUR Sec IV timescape-vs-LCDM
# comparison under it. Settles the sign of the supernova preference the paper currently flags
# as "reproducible, next step" (the central "we can neither reproduce nor refute" caveat).
#
# WHAT IS REPRODUCED
# ------------------
# Seifert/Lane rebuild the Pantheon+ covariance WITHOUT the BBC bias correction (which assumes an
# FLRW fiducial): C = C_fit + C_stat + C_dupl + C_FITOPTS + C_MUOPTS on the (m_B, x1, c) space,
# with f=1 and sigma_floor=0 (see how_to_covariance.ipynb). The finished 5070x5070 (=3N, N=1690)
# covariance PP_1690_COVd.txt and the P+1690 input vector are published at
# Zenodo doi:10.5281/zenodo.12729746 ("P+1690 Covariance Matrix"). BuildPP.py cannot rebuild it
# locally (needs the Pantheon+ .FITRES calibration files, "too large to upload"); we therefore use
# the authors' PUBLISHED covariance artifact -- a more faithful reproduction than a from-scratch
# rebuild that could differ subtly.
#
# LIKELIHOOD (Nielsen-Guffanti-Sarkar / Dam-Heinesen-Wiltshire hierarchical Gaussian; verbatim from
# antosft/SNe-PantheonPlus-Analysis freq_loop.py::GetMaxLikelihood, m2loglike):
#   per SN i, latent SALT2 population means (X0, C0), standardisation (alpha, beta, M0), population
#   variances (V_x, V_c, V_M). Residual 3-vector r_i = (mB_i - mu_i - (M0 - alpha X0 + beta C0),
#   x1_i - X0, c_i - C0). Covariance  C = COVd + blockdiag(B_i),
#   B_i = [[V_M + V_x alpha^2 + V_c beta^2, -V_x alpha, V_c beta],[-V_x alpha, V_x, 0],[V_c beta,0,V_c]].
#   -2 lnL = 3N ln 2pi + ln|C| + r^T C^-1 r.  Cosmology enters ONLY through mu_i.
# Distances: reuse src/fit_timescape.py D_shape_TS / D_shape_LCDM. These are numerically identical
# (ratio constant to 1e-12) to Seifert's own distmod.py timescape/flrw classes (verified), so the
# overall c/H0 * hf normalisation is degenerate with M0 and cancels in the fit. mu enters as
#   mu_shape_i = 5 log10( (1+zHEL_i) * D_shape(zCMB_i, param) )   (offset absorbed by M0).
#
# FAST INNER SOLVE: at fixed cosmology and fixed (alpha,beta,V_x,V_c,V_M) the covariance C is fixed
# and the residual is LINEAR in the offsets p=(M0,X0,C0): r = y - H p, H_i=[[1,-alpha,beta],[0,1,0],
# [0,0,1]], y_i=(mB_i-mu_i, x1_i, c_i). Hence p is solved by 3x3 GLS analytically; the outer
# optimiser only ranges over (alpha, beta, ln V_x, ln V_c, ln V_M).
#
# OUTPUT: probes_out/seifert.json.  DATA (uncommitted, high-volume): set SEIFERT_DATA to the dir
# holding PP_1690_COVd.npy (cached from COVd.txt) + PP_1690_input.txt.  k_TS = k_LCDM = 9 -> dBIC =
# dAIC = Delta(-2lnL). lnB by 1-D quadrature over the cosmological parameter (nuisances profiled;
# identical nuisance structure across models -> Occam factors cancel, same methodology as
# src/probes/evidence.py). Sign: dBIC, lnB > 0 favour LCDM.
import os, sys, json, time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize

WT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(WT, "src")
os.chdir(SRC); sys.path.insert(0, SRC)
import fit_timescape as F

DATA = os.environ.get("SEIFERT_DATA",
    "/private/tmp/claude-501/-Users-s-dev-science-timescape-hubble-tension/"
    "5b0def28-5b41-446e-b96a-22e196fa493b/scratchpad/seifert_data")
REPO = os.environ.get("SEIFERT_REPO",
    "/private/tmp/claude-501/-Users-s-dev-science-timescape-hubble-tension/"
    "5b0def28-5b41-446e-b96a-22e196fa493b/scratchpad/SNe-PantheonPlus-Analysis")
OUTJ = os.path.join(WT, "probes_out", "seifert.json")
QUICK = os.environ.get("SEIFERT_QUICK") == "1"
MAXSEC = float(os.environ.get("SEIFERT_MAXSEC", 0)) or None   # soft wall-clock limit -> clean exit

LN2PI = np.log(2.0 * np.pi)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

def _atomic_dump(obj, path, **kw):
    """Write JSON atomically (temp + rename) so a reaper kill mid-write can't corrupt the file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, **kw)
    os.replace(tmp, path)

# ---------------------------------------------------------------- load ----
def load_covd():
    npy = os.path.join(DATA, "PP_1690_COVd.npy")
    if os.path.exists(npy):
        return np.load(npy)
    C = np.loadtxt(os.path.join(DATA, "PP_1690_COVd.txt"))
    np.save(npy, C)
    return C

log("loading Seifert P+1690 input + 5070x5070 cosmology-independent covariance ...")
INP = np.loadtxt(os.path.join(REPO, "Pantheon", "Build", "PP_1690_input.txt"))
# cols: zCMB, mB, x1, c, HOST_LOGMASS, IDSURVEY, zHEL, RA, DEC
zCMB_all, mB_all, x1_all, c_all, zHEL_all = INP[:,0], INP[:,1], INP[:,2], INP[:,3], INP[:,6]
COVd_all = load_covd()
Nall = len(zCMB_all)
assert COVd_all.shape == (3*Nall, 3*Nall), COVd_all.shape
# sort by zCMB (freq_loop convention), reorder covariance blocks in lockstep
order = np.argsort(zCMB_all)
o3 = np.vstack((3*order, 3*order+1, 3*order+2)).T.ravel()
zCMB, mB, x1, c, zHEL = (a[order] for a in (zCMB_all, mB_all, x1_all, c_all, zHEL_all))
COVd_sorted = COVd_all[np.ix_(o3, o3)]
log(f"N={Nall}; zCMB {zCMB.min():.4f}..{zCMB.max():.4f}; covariance sorted by zCMB")

# ------------------------------------------------------ likelihood core ----
def build_case(zcut):
    """Return everything fixed at a redshift cut: sub-covariance and data blocks."""
    imin = int(np.argmax(zCMB >= zcut))
    sl = slice(imin, None); s3 = slice(3*imin, None)
    Cd = np.ascontiguousarray(COVd_sorted[s3, s3])          # 3n x 3n
    n = Nall - imin
    d = dict(n=n, zcut=float(zcut), imin=imin,
             zCMB=zCMB[sl], zHEL=zHEL[sl], mB=mB[sl], x1=x1[sl], c=c[sl], Cd=Cd)
    d["idx0"] = 3*np.arange(n)                               # block row starts
    return d

def neg2lnL(case, mu_shape, theta):
    """theta = (alpha, beta, lnVx, lnVc, lnVM). mu_shape = per-SN 5log10((1+zHEL)Dshape)."""
    a, b, lvx, lvc, lvm = theta
    Vx, Vc, VM = np.exp(lvx), np.exp(lvc), np.exp(lvm)
    n = case["n"]; i0 = case["idx0"]
    # C = COVd + blockdiag(B_i);  B_i identical across SNe
    M = case["Cd"].copy()
    B = np.array([[VM + Vx*a*a + Vc*b*b, -Vx*a, Vc*b],
                  [-Vx*a,                 Vx,    0.0 ],
                  [ Vc*b,                 0.0,   Vc  ]])
    for p in range(3):
        for q in range(3):
            M[i0+p, i0+q] += B[p, q]
    try:
        L = linalg.cho_factor(M, lower=True, overwrite_a=True, check_finite=False)
    except linalg.LinAlgError:
        return 1e12, None
    logdet = 2.0 * np.sum(np.log(np.diag(L[0])))
    # residual r = y - H p ; solve offsets p=(M0,X0,C0) by GLS
    y = np.empty(3*n)
    y[i0]   = case["mB"] - mu_shape
    y[i0+1] = case["x1"]
    y[i0+2] = case["c"]
    H = np.zeros((3*n, 3))                                   # H_i = [[1,-a,b],[0,1,0],[0,0,1]]
    H[i0,   0] = 1.0; H[i0,   1] = -a; H[i0,   2] = b
    H[i0+1, 1] = 1.0
    H[i0+2, 2] = 1.0
    RHS = np.column_stack([y, H])                            # 3n x 4
    sol = linalg.cho_solve(L, RHS, check_finite=False)       # C^-1 [y|H]
    Ciy, CiH = sol[:, 0], sol[:, 1:]
    A = H.T @ CiH                                            # 3x3
    bvec = H.T @ Ciy                                         # 3
    p = np.linalg.solve(A, bvec)
    quad = float(y @ Ciy - bvec @ p)
    return 3*n*LN2PI + logdet + quad, p

def profile_point(case, mu_shape, x0):
    """Minimise -2lnL over (alpha,beta,lnVx,lnVc,lnVM) at fixed cosmology; warm-start x0.
    Powell chosen after a benchmark: warm ~75 evals/pt vs Nelder-Mead ~194; all optimisers
    (NM/Powell/L-BFGS-B) agree on -2lnL and the nuisances to <1e-3, so the surface is benign."""
    f = lambda th: neg2lnL(case, mu_shape, th)[0]
    r = minimize(f, x0=x0, method="Powell",
                 options=dict(xtol=1e-5, ftol=1e-5, maxiter=8000))
    return r.fun, r.x

def refine_min(grid, y):
    i = int(np.argmin(y))
    if 0 < i < len(grid)-1:
        y1,y2,y3 = y[i-1:i+2]; x1v,x2v,x3v = grid[i-1:i+2]
        dd = y1 - 2*y2 + y3
        if dd > 0:
            xm = x2v + 0.5*(y1-y3)/dd*(x2v-x1v)
            ym = y2 - 0.125*(y1-y3)**2/dd
            return float(xm), float(ym), i
    return float(grid[i]), float(y[i]), i

def dchi2_interval(grid, y):
    ym = y.min(); d = y - ym; i = int(np.argmin(y))
    lo = float(np.interp(1.0, d[:i+1][::-1], grid[:i+1][::-1])) if i>0 and d[0]>=1 else None
    hi = float(np.interp(1.0, d[i:], grid[i:])) if i<len(grid)-1 and d[-1]>=1 else None
    return lo, hi

# initial nuisance guess (alpha,beta from Pantheon+ BBC ballpark; V from freq_loop pre_found)
X0_NUIS = np.array([0.14, 3.10, np.log(0.81), np.log(0.0048), np.log(0.011)])

def _save_raw(parts, pkey, zc, grid, n2, nuis, sols, done):
    parts[pkey] = dict(complete=False, zcut=float(zc), grid=[float(x) for x in grid],
        n2_raw=[float(n2[i]) if done[i] else None for i in range(len(grid))],
        nuis_raw=[[float(x) for x in nuis[i]] if done[i] else None for i in range(len(grid))],
        sols_raw=[[float(x) for x in sols[i]] if (done[i] and sols[i] is not None) else None
                  for i in range(len(grid))])
    _atomic_dump(parts, PARTJ)

def _finalize(grid, n2, nuis, sols):
    g = np.asarray(grid); nn = np.asarray(n2)
    best, n2min, imin = refine_min(g, nn)
    lo, hi = dchi2_interval(g, nn)
    off = sols[imin]; nm = np.asarray(nuis[imin])
    return dict(complete=True, grid=[float(x) for x in grid], n2=[float(x) for x in n2],
        nuis_at_min=[float(x) for x in nm], best=best, n2min=n2min, imin=imin,
        err_lo=(best-lo) if lo else None, err_hi=(hi-best) if hi else None,
        offsets_M0_X0_C0=(None if off is None else [float(x) for x in off]),
        alpha=float(nm[0]), beta=float(nm[1]), Vx=float(np.exp(nm[2])),
        Vc=float(np.exp(nm[3])), VM=float(np.exp(nm[4])))

def profile_model(case, shape_fn, grid, tag, parts, pkey):
    """Nuisance-optimised profile over the cosmological grid, with INTRA-grid checkpointing:
    saves every 3 points, so the harness's periodic background-task reaper costs at most a few
    points and a relaunch resumes the unfinished ones. Returns a finalised (complete) profile."""
    prior = parts.get(pkey)
    if prior and prior.get("complete"):
        return prior
    n = len(grid)
    n2 = np.full(n, np.nan); nuis = np.full((n, 5), np.nan); sols = [None]*n
    done = np.zeros(n, bool)
    if prior and prior.get("n2_raw"):                       # resume an incomplete partial
        for i in range(n):
            v = prior["n2_raw"][i]
            if v is not None:
                n2[i] = v; nuis[i] = prior["nuis_raw"][i]; done[i] = True
                sr = prior["sols_raw"][i]
                sols[i] = np.array(sr) if sr is not None else None
        log(f"  {tag} z>={case['zcut']:.3f}: resuming ({int(done.sum())}/{n} points)")
    mus = [5.0*np.log10((1.0+case["zHEL"]) * shape_fn(g)) for g in grid]
    i0 = n//2
    order_scan = list(range(i0, n)) + list(range(i0-1, -1, -1))
    def warm(idx):
        for d in range(1, n):
            for j in (idx-d, idx+d):
                if 0 <= j < n and done[j]: return nuis[j]
        return X0_NUIS
    since = 0
    for idx in order_scan:
        if done[idx]: continue
        fun, xopt = profile_point(case, mus[idx], warm(idx))
        n2[idx] = fun; nuis[idx] = xopt; _, sols[idx] = neg2lnL(case, mus[idx], xopt); done[idx] = True
        since += 1
        if since >= 3:
            _save_raw(parts, pkey, case["zcut"], grid, n2, nuis, sols, done); since = 0
            if MAXSEC and time.time()-t0 > MAXSEC:          # soft limit -> clean exit for relaunch
                log(f"  {tag} z>={case['zcut']:.3f}: soft time limit at {int(done.sum())}/{n} pts; exiting")
                sys.exit(0)
    fin = _finalize(grid, n2, nuis, sols)
    parts[pkey] = fin
    _atomic_dump(parts, PARTJ)
    log(f"  {tag} z>={case['zcut']:.3f} (N={case['n']}): best={fin['best']:.4f} "
        f"-2lnL={fin['n2min']:.3f} err=+{fin['err_hi'] or float('nan'):.4f}/-{fin['err_lo'] or float('nan'):.4f}")
    return fin

# ------------------------------------------------------------- lnB ---------
def logtrapz(lnf, x):
    m = float(np.max(lnf)); return m + float(np.log(np.trapezoid(np.exp(lnf-m), x)))

def lnZ_hat(grid, n2, lo, hi, cref):
    g = np.asarray(grid); msk = (g>=lo-1e-12)&(g<=hi+1e-12)
    gg = g[msk]; cc = np.asarray(n2)[msk]
    return logtrapz(-0.5*(cc-cref), gg) - float(np.log(hi-lo))

# ------------------------------------------------------------- run ---------
# grids: step ~0.02; each profile point is nuisance-optimised, min parabola-refined, so
# dBIC / fv0 / Delta-chi2=1 errors are accurate sub-grid. Range spans the lnB prior support.
FV_GRID = np.linspace(0.50, 0.98, int(os.environ.get("SEIFERT_NFV", 13 if QUICK else 25)))  # step 0.02
OM_GRID = np.linspace(0.06, 0.60, int(os.environ.get("SEIFERT_NOM", 13 if QUICK else 28)))  # step 0.02
FV_PRIORS = {"fv0~U(0.50,0.95)": (0.50, 0.95), "fv0~U(0.55,0.90)": (0.55, 0.90)}
OM_PRIORS = {"Om~U(0.06,0.60)": (0.06, 0.60), "Om~U(0.10,0.50)": (0.10, 0.50)}
ZCUTS = [0.0, 0.055] if QUICK else [0.0, 0.033, 0.055, 0.075, 0.10]
if os.environ.get("SEIFERT_ZCUTS"):                        # override e.g. "0.0,0.033" for resume
    ZCUTS = [float(x) for x in os.environ["SEIFERT_ZCUTS"].split(",")]

PROVENANCE = dict(
    worktree=WT, script="src/probes/seifert.py",
    covariance="Seifert et al. cosmology-independent P+1690 covariance, Zenodo doi:10.5281/zenodo.12729746 "
               "(PP_1690_COVd.txt, 5070x5070); input PP_1690_input.txt from antosft/SNe-PantheonPlus-Analysis "
               "@3caa94c (byte-identical to the Zenodo full_input.csv)",
    likelihood="NGS/Dam hierarchical Gaussian, verbatim from freq_loop.py::m2loglike; distances reuse "
               "src/fit_timescape.py (verified identical to Seifert distmod.py to 1e-12); likelihood cross-"
               "checked to 1e-12 against an independent slogdet+solve implementation",
    k="9 params each model (cosmo + alpha,beta,M0,X0,C0,V_x,V_c,V_M) -> dBIC = dAIC = Delta(-2lnL)",
    lnB="1-D quadrature over the cosmological parameter, nuisances profiled (Occam factors cancel across "
        "models: identical nuisance structure); primary decisive statistic is dBIC",
    sign="dBIC, lnB > 0 favour LCDM; < 0 favour timescape",
    data_env="SEIFERT_DATA / SEIFERT_REPO (uncommitted high-volume data; not in repo)")

def checkpoint(results):
    """Incremental dump after each zcut so a kill on this shared box preserves completed cuts.
    Merges into any existing seifert.json (resume-friendly)."""
    prior = {}
    if os.path.exists(OUTJ):
        try:
            prior = json.load(open(OUTJ)).get("results", {})
        except Exception:
            prior = {}
    prior.update(results)
    seif = prior.get("z>=0.055", {})
    gate = dict(seifert_published=dict(fv0=0.737, err=0.029, zmin=0.055,
                    source="arXiv:2412.15143 (MNRAS Lett. 537, L55)"),
                our_fv0_z055=seif.get("timescape", {}).get("fv0"),
                deviation_sigma=(abs(seif.get("timescape",{}).get("fv0", np.nan)-0.737)/0.029
                                 if seif.get("timescape") else None))
    out = dict(name="seifert", provenance=PROVENANCE,
               fv_grid=[float(FV_GRID[0]), float(FV_GRID[-1]), len(FV_GRID)],
               om_grid=[float(OM_GRID[0]), float(OM_GRID[-1]), len(OM_GRID)],
               results=prior, literature_gate=gate, runtime_s=round(time.time()-t0, 1))
    _atomic_dump(out, OUTJ, indent=2)
    return out

# per-(model,zcut) partials so each chunk stays under the harness's ~20-min background-task limit;
# a relaunch skips any (model,zcut) already stored and resumes the rest.
PARTJ  = os.path.join(WT, "probes_out", "seifert_partials.json")
MODELS = os.environ.get("SEIFERT_MODEL", "both")           # both | TS | LCDM

def load_partials():
    if os.path.exists(PARTJ):
        try: return json.load(open(PARTJ))
        except Exception: return {}
    return {}

def save_partial(parts, key, prof):
    parts[key] = prof
    _atomic_dump(parts, PARTJ)

def combine(zc, N, ts, lc):
    dbic = ts["n2min"] - lc["n2min"]                        # k equal
    cref = min(ts["n2min"], lc["n2min"])
    lnB = {}
    for po,(al,ah) in OM_PRIORS.items():
        zL = lnZ_hat(lc["grid"], lc["n2"], al, ah, cref)
        for pf,(fl,fh) in FV_PRIORS.items():
            zT = lnZ_hat(ts["grid"], ts["n2"], fl, fh, cref)
            lnB[f"{po} x {pf}"] = dict(lnB_LCDM_over_TS=zL-zT, twolnB=2*(zL-zT))
    fv = ts["best"]; om_ts = 0.5*(1-fv)*(2+fv)
    return dict(zcut=zc, N=N,
        timescape=dict(fv0=fv, err_lo=ts["err_lo"], err_hi=ts["err_hi"], neg2lnL=ts["n2min"],
                       Om_dressed_implied=om_ts, alpha=ts["alpha"], beta=ts["beta"],
                       Vx=ts["Vx"], Vc=ts["Vc"], VM=ts["VM"], nuis_at_min=ts["nuis_at_min"]),
        lcdm=dict(Om=lc["best"], err_lo=lc["err_lo"], err_hi=lc["err_hi"], neg2lnL=lc["n2min"],
                  alpha=lc["alpha"], beta=lc["beta"], Vx=lc["Vx"], Vc=lc["Vc"], VM=lc["VM"]),
        dBIC_TS_minus_LCDM=dbic, dAIC_TS_minus_LCDM=dbic, lnB=lnB,
        ts_curve=dict(grid=ts["grid"], n2=ts["n2"]),
        lcdm_curve=dict(grid=lc["grid"], n2=lc["n2"]))

def main():
  parts = load_partials()
  for zc in ZCUTS:
    zkey = f"z>={zc:.3f}"; case = None
    def ensure(model, shape_fn, grid, tag):
        nonlocal case
        k = f"{zkey}:{model}"
        prior = parts.get(k)
        if prior and prior.get("complete"): return prior
        if MODELS not in ("both", model): return None       # not requested this run
        if case is None: case = build_case(zc)
        return profile_model(case, shape_fn, grid, tag, parts, k)  # resumes + checkpoints internally
    ts = ensure("TS",   lambda fv: F.D_shape_TS(case["zCMB"], fv),  FV_GRID, "TS  ")
    lc = ensure("LCDM", lambda om: F.D_shape_LCDM(case["zCMB"], om), OM_GRID, "LCDM")
    if ts is not None and lc is not None:
        N = case["n"] if case is not None else build_case(zc)["n"]
        res = combine(zc, N, ts, lc)
        log(f"== {zkey}: fv0={res['timescape']['fv0']:.4f}"
            f"(+{ts['err_hi'] or float('nan'):.4f}/-{ts['err_lo'] or float('nan'):.4f}) "
            f"Om={res['lcdm']['Om']:.4f}  dBIC(TS-LCDM)={res['dBIC_TS_minus_LCDM']:+.2f}  "
            f"lnB={list(res['lnB'].values())[0]['lnB_LCDM_over_TS']:+.2f}")
        checkpoint({zkey: res})                              # persist this cut immediately

  out = checkpoint({})   # final merged dump (+ literature gate)
  log(f"wrote {OUTJ}")
  print("\n=== SUMMARY (dBIC>0 favours LCDM) ===")
  for k, v in out["results"].items():
      lnb0 = list(v["lnB"].values())[0]["lnB_LCDM_over_TS"]
      print(f"  {k}: N={v['N']:4d} fv0={v['timescape']['fv0']:.4f} Om={v['lcdm']['Om']:.4f} "
            f"dBIC={v['dBIC_TS_minus_LCDM']:+7.2f} lnB={lnb0:+6.2f}")
  g = out["literature_gate"]
  print(f"\nLiterature gate (z>=0.055 vs Seifert 0.737+-0.029): our fv0={g['our_fv0_z055']}, "
        f"dev={g['deviation_sigma']:.2f}sigma" if g['our_fv0_z055'] else "gate n/a")
  return out

if __name__ == "__main__":
    main()
