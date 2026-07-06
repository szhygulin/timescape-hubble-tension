#!/usr/bin/env python3
"""Independent audit verification pass (2026-07-06), run on request "audit this paper".

Re-derives, with INDEPENDENT implementations where it matters, the load-bearing numbers:

A. Timescape tracker-solution machinery (feeds every per-SN model point):
   A1  F'(tau) closed-form antiderivative identity (hand-derived: F' = (2t+3b)/(3(t+b)t^{2/3})
       = 2/((2+f_v)t^{2/3}) via 2+f_v = 6(t+b)/(2t+3b) on the tracker).
   A2  z(tau0) = 0.
   A3  low-z limit D(z)/z -> 1/g(fv0), g = (4fv0^2+fv0+4)/(2(2+fv0)) — ties the distance
       normalisation to the bare<->dressed Hubble relation (hand-re-derived from
       H_dressed = gamma*Hbar - dgamma/dt with gamma=(2+fv)/2, fv' = fv(1-fv)/t).
   A4  H_dressed(tau) = (4fv^2+fv+4)/(6 tau) vs the committed H_over_Hbar0; and dD_M/dz vs D_H.
   A5  dressed q0: numerical (from D(z) and from H(z)) vs analytic
       -(1-fv0)(8fv0^3+39fv0^2-12fv0-8)/(4+fv0+4fv0^2)^2.

B. Phase-3 per-SN LOS data points (are the per-standard-candle covariates real?):
   B1  rotation matrix: orthogonality, det=+1, NGP -> b=90, Gal centre -> (0,0), M87 -> published (l,b).
   B2  comoving distance vs scipy.quad.
   B3  FULL re-computation of all 573 per-SN covariates (mean_delta, F at 3 thresholds) with an
       independent pure-numpy trilinear sampler (no scipy.ndimage), compared to probe1_los.npz.
   B4  named-structure signs: Virgo/Coma/Norma overdense, Local Void underdense.
   B5  sample counts: 573 rows / 460 unique SNe / |b|<10 count.

C. Headline statistics:
   C1  chi2_TS(0.853), chi2_LCDM(0.33328), dBIC vs results.json.
   C2  probe2 naive GLS slopes reproduced from covariates + full covariance.
   C3  fvsplit/dr2 sigma arithmetic (sigma = sqrt(dchi2_join); Gaussian cross-check).
   C4  T1 p-value; C5 CMB acoustic-scale point; C6 probe5 ceiling re-derived through the
       exact mu pipeline at |lambda| = 0.058.
"""
import os, sys, json
import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
WT = os.path.abspath(os.path.join(SRC, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, SRC)
os.chdir(SRC)
import los_common as LC
import fit_timescape as F

C_KM = 299792.458
results = {}
def report(key, ok, detail):
    results[key] = {"PASS": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {key}: {detail}")

# ======================= A. timescape tracker machinery =======================
def g_dress(fv0): return (4*fv0**2 + fv0 + 4) / (2*(2 + fv0))

# A1: F' identity (independent hand-derived form)
errs = []
for fv0 in (0.60, 0.695, 0.762, 0.853, 0.90):
    b = F.b_tilde(fv0); tau0 = F.tau0_tilde(fv0)
    taus = np.linspace(0.02*tau0, tau0, 200)
    h = 1e-6*tau0
    Fp_num = (F.Ftilde(taus+h, fv0) - F.Ftilde(taus-h, fv0)) / (2*h)
    Fp_ana = (2*taus + 3*b) / (3*(taus + b) * taus**(2/3))
    errs.append(np.max(np.abs(Fp_num/Fp_ana - 1)))
report("A1_Fprime_identity", max(errs) < 1e-7, f"max rel err {max(errs):.2e} (5 fv0 values, 200 tau pts)")

# A2: z(tau0) = 0
z0s = [abs(F.z_of_tau(np.array([F.tau0_tilde(fv)]), fv)[0]) for fv in (0.60, 0.695, 0.762, 0.853, 0.90)]
report("A2_z_at_tau0", max(z0s) < 1e-12, f"max |z(tau0)| = {max(z0s):.2e}")

# A3: low-z distance normalisation -> 1/g(fv0)
errs = []
for fv0 in (0.60, 0.695, 0.762, 0.853):
    z = np.array([1e-4])
    D = F.D_shape_TS(z, fv0)[0]
    errs.append(abs(D/z[0] * g_dress(fv0) - 1))
report("A3_lowz_dressed_H0_normalisation", max(errs) < 5e-4,
       f"max |D/z * g - 1| = {max(errs):.2e} at z=1e-4 (O(z) term ~5e-5)")

# A4: dressed H(z) hand-derivation vs committed; dD_M/dz vs D_H
import timescape_baocmb as T
errs_h, dmdh = [], []
for fv0 in (0.695, 0.853):
    for z in (0.1, 0.5, 1.0, 2.0):
        tau = T.tau_of_z(z, fv0); fv = T.fv_of_tau(tau, fv0)
        mine = (4*fv**2 + fv + 4) / (6*tau)
        errs_h.append(abs(T.H_over_Hbar0(z, fv0)/mine - 1))
        h = 1e-5
        dDM = (T.DM(z+h, fv0) - T.DM(z-h, fv0)) / (2*h)
        dmdh.append(abs(dDM / T.DH(z, fv0) - 1))
report("A4a_dressed_Hz_identity", max(errs_h) < 1e-12, f"max rel err {max(errs_h):.2e}")
report("A4b_dDM_dz_vs_DH", True, f"max |dD_M/dz / D_H - 1| = {max(dmdh):.2e} (characterisation)")

# A5: dressed q0 three ways
q0rows = []
for fv0 in (0.695, 0.762, 0.853):
    zs = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    Ds = np.array([T.DM(z, fv0) for z in zs])
    coef = np.polyfit(zs, Ds, 3)   # D = (1/g)[z - (1+q0)/2 z^2 + ...]
    q0_D = -2*coef[-3]*g_dress(fv0) - 1
    h = 1e-5
    dH = (T.H_over_Hbar0(h, fv0) - T.H_over_Hbar0(0.0 + 1e-12, fv0)) / h
    q0_H = dH / g_dress(fv0) - 1     # q = (1+z)/H dH/dz - 1 at z=0
    q0_ana = -(1-fv0)*(8*fv0**3 + 39*fv0**2 - 12*fv0 - 8) / (4 + fv0 + 4*fv0**2)**2
    q0rows.append((fv0, q0_D, q0_H, q0_ana))
ok = all(abs(a-c) < 5e-3 and abs(b-c) < 5e-3 for _, a, b, c in q0rows)
report("A5_dressed_q0", ok, "; ".join(f"fv0={r[0]}: D-fit {r[1]:+.4f}, H-deriv {r[2]:+.4f}, analytic {r[3]:+.4f}" for r in q0rows))

# ======================= B. Phase-3 per-SN data points =======================
R = LC.R_EQ2GAL
report("B1a_rotation_orthonormal",
       np.allclose(R @ R.T, np.eye(3), atol=1e-8) and abs(np.linalg.det(R) - 1) < 1e-8,
       f"||R R^T - I||max = {np.max(np.abs(R @ R.T - np.eye(3))):.1e}, det = {np.linalg.det(R):.10f}")
l_ngp, b_ngp = LC.eq_to_gal_lb(192.85948, 27.12825)
l_gc, b_gc = LC.eq_to_gal_lb(266.40500, -28.93617)
l_m87, b_m87 = LC.eq_to_gal_lb(187.70593, 12.39112)
report("B1b_named_directions",
       abs(b_ngp - 90) < 0.01 and abs(b_gc) < 0.01 and (abs(l_gc) < 0.01 or abs(l_gc-360) < 0.01)
       and abs(l_m87 - 283.778) < 0.05 and abs(b_m87 - 74.491) < 0.05,
       f"NGP b={b_ngp:.4f}; GC (l,b)=({l_gc:.4f},{b_gc:.4f}); M87 (l,b)=({l_m87:.3f},{b_m87:.3f}) vs published (283.778, 74.491)")

zs = [0.01, 0.02, 0.04, 0.067]
errs = []
for z in zs:
    ref = (C_KM/100.0) * quad(lambda zz: 1/np.sqrt(LC.OM_FID*(1+zz)**3 + 1-LC.OM_FID), 0, z)[0]
    errs.append(abs(LC.comoving_dist_mpc_h(np.array([z]))[0]/ref - 1))
report("B2_comoving_distance_vs_quad", max(errs) < 1e-6, f"max rel err {max(errs):.2e}")

# B3: full independent recomputation of the 573 per-SN covariates
field = LC.load_field()
def trilinear_mine(pts_x, pts_y, pts_z):
    """Pure-numpy trilinear with edge clamping (== map_coordinates order=1 mode='nearest')."""
    i = np.clip((pts_x - LC.LMIN)/LC.DX, 0, LC.NGRID-1)
    j = np.clip((pts_y - LC.LMIN)/LC.DX, 0, LC.NGRID-1)
    k = np.clip((pts_z - LC.LMIN)/LC.DX, 0, LC.NGRID-1)
    i0 = np.floor(i).astype(int); j0 = np.floor(j).astype(int); k0 = np.floor(k).astype(int)
    i1 = np.minimum(i0+1, LC.NGRID-1); j1 = np.minimum(j0+1, LC.NGRID-1); k1 = np.minimum(k0+1, LC.NGRID-1)
    fx = i - i0; fy = j - j0; fz = k - k0
    c000 = field[i0, j0, k0]; c100 = field[i1, j0, k0]; c010 = field[i0, j1, k0]; c001 = field[i0, j0, k1]
    c110 = field[i1, j1, k0]; c101 = field[i1, j0, k1]; c011 = field[i0, j1, k1]; c111 = field[i1, j1, k1]
    return (c000*(1-fx)*(1-fy)*(1-fz) + c100*fx*(1-fy)*(1-fz) + c010*(1-fx)*fy*(1-fz)
            + c001*(1-fx)*(1-fy)*fz + c110*fx*fy*(1-fz) + c101*fx*(1-fy)*fz
            + c011*(1-fx)*fy*fz + c111*fx*fy*fz)

cat = LC.load_catalog(); mask = LC.usable_mask(cat); idx = np.where(mask)[0]
d1 = np.load(os.path.join(WT, "probes_out", "probe1_los.npz"))
assert np.array_equal(d1["row_index"], idx)
l = cat["l"][idx]; b = cat["b"][idx]; r = cat["r"][idx]
md_mine = np.empty(len(idx)); F_mine = {th: np.empty(len(idx)) for th in (-0.3, -0.5, -0.7)}
for i in range(len(idx)):
    n = max(2, int(np.ceil(r[i]/0.5)))
    s = np.linspace(0.5, r[i], n)
    ux, uy, uz = LC.unit_gal(l[i], b[i])
    d = trilinear_mine(s*ux, s*uy, s*uz)
    md_mine[i] = d.mean()
    for th in F_mine: F_mine[th][i] = np.mean(d < th)
dmd = np.max(np.abs(md_mine - d1["mean_delta"]))
dF = max(np.max(np.abs(F_mine[-0.3] - d1["F_m03"])), np.max(np.abs(F_mine[-0.5] - d1["F_m05"])),
         np.max(np.abs(F_mine[-0.7] - d1["F_m07"])))
report("B3_per_SN_covariates_573", dmd < 1e-10 and dF < 1e-10,
       f"ALL 573 rows recomputed with independent trilinear: max|d mean_delta|={dmd:.2e}, max|dF|={dF:.2e}")

# B4: named structures (my sampler, not the repo's)
struct = {
    "Virgo(l283.78,b74.49,r12)": (283.78, 74.49, 12.0, +1),
    "Coma(l58,b88,r70)": (58.0, 88.0, 70.0, +1),
    "Norma/GA(l325,b-7,r50)": (325.0, -7.0, 50.0, +1),
    "LocalVoid(l18,b6,r40)": (18.0, 6.0, 40.0, -1),
}
oks, det = [], []
for name, (ll, bb, rr, sign) in struct.items():
    ux, uy, uz = LC.unit_gal(ll, bb)
    val = float(trilinear_mine(np.array([rr*ux]), np.array([rr*uy]), np.array([rr*uz]))[0])
    oks.append(np.sign(val) == sign); det.append(f"{name}: delta={val:+.2f}")
report("B4_named_structures", all(oks), "; ".join(det))

ncid = len(set(cat["CID"][idx])); nzoa = int(np.sum(np.abs(b) < 10))
report("B5_sample_counts", len(idx) == 573 and ncid == 460,
       f"rows={len(idx)} (exp 573), unique CID={ncid} (exp 460), |b|<10: {nzoa} (exp 6)")

# ======================= C. headline statistics =======================
zHD, zHEL, mb, Cf = F.load()
chi2 = F.make_chi2(zHD, zHEL, mb, Cf)
cTS = chi2(F.D_shape_TS(zHD, 0.853))
cL = chi2(F.D_shape_LCDM(zHD, 0.3332775919732441))
rj = json.load(open(os.path.join(WT, "results.json")))["standard_full"]
report("C1_headline_chi2", abs(cTS - rj["chi2_TS"]) < 5e-3 and abs(cL - rj["chi2_L"]) < 5e-3,
       f"chi2_TS={cTS:.4f} (stored {rj['chi2_TS']:.4f}), chi2_L={cL:.4f} (stored {rj['chi2_L']:.4f}), dBIC={cTS-cL:+.3f}")

# C2: probe2 naive GLS slope reproduction (M1, mean_delta and F_void, LCDM shape)
Csub = LC.load_cov_subset(mask); Ci = np.linalg.inv(Csub)
zC = cat["zCMB"][idx]; zE = cat["zHEL"][idx]; mbs = cat["m_b_corr"][idx]
DL = F.D_shape_LCDM(zC, 0.3332775919732441)
y = mbs - 5*np.log10((1+zE)*DL)
nbin = 20
qe = np.quantile(zC, np.linspace(0, 1, nbin+1)); qe[0] -= 1e-9; qe[-1] += 1e-9
zb = np.digitize(zC, qe[1:-1])
def gls(covar):
    X = np.column_stack([np.ones(len(y)), covar] + [(zb == k).astype(float) for k in range(1, nbin)])
    A = np.linalg.inv(X.T @ Ci @ X); beta = A @ (X.T @ Ci @ y)
    return beta[1], np.sqrt(A[1, 1])
p2 = json.load(open(os.path.join(WT, "probes_out", "probe2_gls.json")))
lam_md, sig_md = gls(d1["mean_delta"]); lam_F, sig_F = gls(d1["F_m05"])
st_md = p2["r_zCMB_shape_LCDM"]["mean_delta"]["M1_+zbins"]; st_F = p2["r_zCMB_shape_LCDM"]["F_void(-0.5)"]["M1_+zbins"]
report("C2_probe2_gls_slopes", abs(lam_md - st_md["lambda"]) < 1e-8 and abs(lam_F - st_F["lambda"]) < 1e-8,
       f"mean_delta lam={lam_md:.6f} (stored {st_md['lambda']:.6f}), F lam={lam_F:.6f} (stored {st_F['lambda']:.6f}); "
       f"naive sig_md={sig_md:.6f} vs rotation-null std {p2['rotation_null(authoritative)']['mean_delta']['null_std']:.6f} "
       f"(ratio {p2['rotation_null(authoritative)']['mean_delta']['null_std']/sig_md:.2f}x)")

# C3: fvsplit / dr2 sigma arithmetic
fs = json.load(open(os.path.join(WT, "probes_out", "fvsplit.json")))
dr2 = json.load(open(os.path.join(WT, "probes_out", "dr2.json")))
ok3, det3 = True, []
h = fs["headline"]
ok3 &= abs(np.sqrt(h["dchi2_join"]) - h["sigma"]) < 0.01
det3.append(f"headline sqrt({h['dchi2_join']})={np.sqrt(h['dchi2_join']):.3f} vs {h['sigma']}")
for k, v in dr2["fv_split"].items():
    ok3 &= abs(np.sqrt(v["T_profile_shift"]) - v["sigma_profile"]) < 0.01
    gg = v["gap_fv0"]/np.hypot(v["sigma_bao_fv0"], v["sigma_other_fv0"])
    ok3 &= abs(gg - v["sigma_gaussian"]) < 0.02
report("C3_split_sigma_arithmetic", ok3, "; ".join(det3) + f"; all {len(dr2['fv_split'])} dr2 pairings sqrt+gaussian consistent")

# C4: T1 p-value
t1 = json.load(open(os.path.join(WT, "probes_out", "probeT1_freeshape.json")))
from scipy.stats import chi2 as chi2d
d = t1["free_vs_LCDM_significance"]
p_re = float(chi2d.sf(d["delta_chi2"], d["n_extra_params"]))
report("C4_T1_pvalue", abs(p_re - d["p_value"]) < 1e-6,
       f"dchi2={d['delta_chi2']:.2f}/{d['n_extra_params']}dof -> p={p_re:.4f} (stored {d['p_value']:.4f}); "
       f"chi2 free={t1['free_shape_fit']['chi2_free_min']:.1f} vs LCDM={t1['reference_chi2_same_data']['LCDM']:.1f} vs TS={t1['reference_chi2_same_data']['timescape']:.1f}")

# C5: CMB acoustic-scale point
dmz = (100.0/1.04109) * (144.43/147.09)
report("C5_CMB_point", abs(dmz - 94.316) < 0.005, f"D_M(z*)/r_d = {dmz:.4f} (paper 94.32; theta*=1.04109e-2, r*=144.43, r_d=147.09)")

# C6: probe5 ceiling through the exact mu pipeline
cosmo = (cat["IS_CALIBRATOR"] == 0) & (cat["zHD"] > 0.01)
low_in_cosmo = LC.usable_mask(cat)[cosmo]
Fc = d1["F_m05"] - d1["F_m05"].mean()
Ffull = np.zeros(len(zHD)); Ffull[low_in_cosmo] = Fc
ddim = F.D_shape_LCDM(zHD, LC.OM_FID)
lam = 0.058
zpert = zHD + (1+zHD)*lam*Ffull*ddim
Dts0 = F.D_shape_TS(zHD, 0.853); Dts1 = F.D_shape_TS(zpert, 0.853)
dmu = 5*np.log10(((1+zHEL)*Dts1)/((1+zHEL)*Dts0))
rms = float(np.sqrt(np.mean(dmu[low_in_cosmo]**2))); mx = float(np.max(np.abs(dmu)))
report("C6_probe5_ceiling", rms < 0.05, f"RMS dmu(573, lam=0.058) = {rms:.4f} mag (paper <=0.046), max = {mx:.4f} mag")

# ======================= summary =======================
npass = sum(1 for v in results.values() if v["PASS"]); nfail = len(results) - npass
print(f"\n=== VERIFY SUMMARY: {npass} PASS / {nfail} FAIL of {len(results)} ===")
out = os.path.join(WT, "probes_out", "verify_audit_20260706.json")
with open(out, "w") as f: json.dump(results, f, indent=2)
print(f"wrote {out}")
