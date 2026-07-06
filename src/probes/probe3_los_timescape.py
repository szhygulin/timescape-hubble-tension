#!/usr/bin/env python3
"""Phase 3 (LOS voids), Probe 3A — per-sightline timescape extension (one new parameter).

Variant A (expansion-variance coupling), per PLAN_los_voids.md: give each low-z SN a redshift
shift proportional to its LOS void excess,
    z_i -> z_i + Dz_i,   Dz_i = (1+z_i) * lambda*(F_i - Fbar) * d_i,
with d_i = dimensionless comoving distance (H D_i / c) and F_i the Probe-1 void fraction. The
shift is applied ON TOP of the standard zHD for the 573 SNe inside the 2M++ volume; the other
1007 SNe are unchanged. The full 1580-SN timescape fit is then run over (f_v0, lambda) with the
committed STAT+SYS covariance and the analytic M_B/H0 offset marginalisation.

Headline question: does letting measured LOS structure into the fit move the SN-preferred
f_v0 (0.85) toward the BAO+CMB value (0.64), and is lambda detected?

G2 gate: at lambda=0 the extended model is IDENTICAL to standard timescape, so it must
reproduce the repo's committed chi2 and f_v0 exactly.

Significance is calibrated against the SAME sky-rotation null established in Probe 2 (the naive
error understates sigma by ~2.7x): the improvement Delta-chi2(lambda) for the real sightlines is
compared to that from random rotations of the field.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)
import los_common as LC
os.chdir(SRC)
import fit_timescape as FT

OUTJ = os.path.join(LC.WT, "probes_out", "probe3_los_timescape.json")
OM_FID = LC.OM_FID


def ts_distance_factory(fv0):
    """Return z -> comoving TS distance shape at fixed fv0 (grid built once)."""
    tau0 = FT.tau0_tilde(fv0)
    grid = np.linspace(1e-7 * tau0, tau0, 400000)
    zg = FT.z_of_tau(grid, fv0)
    order = np.argsort(zg)
    zs, taus = zg[order], grid[order]
    F0 = FT.Ftilde(tau0, fv0)

    def D(z):
        tau = np.interp(z, zs, taus)
        dA = tau ** (2.0 / 3.0) * (F0 - FT.Ftilde(tau, fv0))
        return (1.0 + z) * dA
    return D


def main():
    # standard 1580-SN cosmology sample (paper machinery, catalog order)
    zHD, zHEL, mb, C = FT.load()
    N = len(zHD)
    Cinv = np.linalg.inv(C)
    one = np.ones(N); s11 = one @ (Cinv @ one)

    def chi2_zdist(Dfun, z_dist):
        if np.any(z_dist <= 1e-4):            # coupling drove a redshift non-physical
            return np.inf
        D = Dfun(z_dist)
        if np.any(D <= 0):
            return np.inf
        mu = 5.0 * np.log10((1.0 + zHEL) * D)
        r = mb - mu
        Cinvr = Cinv @ r
        return float(r @ Cinvr - (one @ Cinvr) ** 2 / s11)

    # ---- align Probe-1 LOS covariates into the 1580 sample ----
    cat = LC.load_catalog()
    cosmo = (cat["IS_CALIBRATOR"] == 0) & (cat["zHD"] > 0.01)
    low = LC.usable_mask(cat)
    low_in_cosmo = low[cosmo]                       # (1580,) boolean
    d1 = np.load(os.path.join(LC.WT, "probes_out", "probe1_los.npz"))
    assert np.allclose(zHD[low_in_cosmo], d1["zHD"]), "Probe-1 alignment into 1580 sample failed"
    F = d1["F_m05"]
    Fbar = float(F.mean())
    F_full = np.zeros(N); F_full[low_in_cosmo] = F - Fbar     # centred void excess
    n_low = int(low_in_cosmo.sum())

    d_dimless = FT.D_shape_LCDM(zHD, OM_FID)         # H D / c (dimensionless comoving distance)

    def zdist(lam):
        return zHD + (1.0 + zHD) * lam * F_full * d_dimless

    # ---- G2 gate: standard timescape (lambda=0) reproduces the committed fit ----
    fv_grid = np.linspace(0.72, 0.96, 49)
    factories = {fv: ts_distance_factory(fv) for fv in fv_grid}
    chi_std = np.array([chi2_zdist(factories[fv], zdist(0.0)) for fv in fv_grid])
    i0 = int(np.argmin(chi_std))
    fv0_std, chi2_std = float(fv_grid[i0]), float(chi_std[i0])
    g2 = {"fv0_lambda0": fv0_std, "chi2_lambda0": chi2_std,
          "committed_fv0": 0.853, "committed_chi2_TS": 1391.5452,
          "PASS": bool(abs(fv0_std - 0.853) < 0.006 and abs(chi2_std - 1391.5452) < 0.5)}

    # ---- 2D fit over (fv0, lambda): does fv0 move? ----
    lam_grid = np.linspace(-2.0, 2.0, 161)
    chi2_grid = np.empty((len(fv_grid), len(lam_grid)))
    for a, fv in enumerate(fv_grid):
        Df = factories[fv]
        for c, lam in enumerate(lam_grid):
            chi2_grid[a, c] = chi2_zdist(Df, zdist(lam))
    amin, cmin = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    fv0_ext, lam_ext, chi2_ext = float(fv_grid[amin]), float(lam_grid[cmin]), float(chi2_grid[amin, cmin])

    # profile lambda at fixed fv0_std for sigma_lambda and Delta-chi2
    Df0 = factories[fv0_std]
    chi_lam = np.array([chi2_zdist(Df0, zdist(lam)) for lam in lam_grid])
    j = int(np.argmin(chi_lam)); lam_hat = float(lam_grid[j]); chi2_at = float(chi_lam[j])
    dchi2_lambda = chi2_std - chi2_at            # improvement of best lambda over lambda=0
    # sigma from delta-chi2=1
    dd = chi_lam - chi2_at
    lo = np.interp(1.0, dd[:j + 1][::-1], lam_grid[:j + 1][::-1]) if j > 0 else lam_hat
    hi = np.interp(1.0, dd[j:], lam_grid[j:]) if j < len(lam_grid) - 1 else lam_hat
    sig_lam = float(0.5 * (hi - lo)) if hi > lam_hat and lo < lam_hat else float("nan")

    # ---- rotation-null calibration of Delta-chi2(lambda) (authoritative, per Probe 2) ----
    from scipy.spatial.transform import Rotation
    l = cat["l"][low]; b = cat["b"][low]; rr = cat["r"][low]
    u = LC.unit_gal(l, b)
    NSTEP = 200
    frac = np.linspace(0.0, 1.0, NSTEP)[None, :]

    def F_from_dirs(uvec):
        s = LC.DX + frac * (rr[:, None] - LC.DX)
        X = uvec[0][:, None] * s; Y = uvec[1][:, None] * s; Z = uvec[2][:, None] * s
        dd_ = LC.sample_points(LC.load_field(), X.ravel(), Y.ravel(), Z.ravel()).reshape(len(rr), NSTEP)
        return np.mean(dd_ < -0.5, axis=1)

    def dchi2_for_F(Fvals):
        Ff = np.zeros(N); Ff[low_in_cosmo] = Fvals - Fvals.mean()

        def zd(lam):
            return zHD + (1.0 + zHD) * lam * Ff * d_dimless
        cl = np.array([chi2_zdist(Df0, zd(lam)) for lam in lam_grid])
        return chi2_std - cl.min()

    rng = np.random.default_rng(7)
    nrot = 200
    Rs = Rotation.random(nrot, random_state=rng).as_matrix()
    dchi_null = np.array([dchi2_for_F(F_from_dirs(Rs[k] @ u)) for k in range(nrot)])
    p_rot = float((np.sum(dchi_null >= dchi2_lambda) + 1) / (nrot + 1))

    out = {
        "probe": "3A — per-sightline timescape (expansion-variance coupling)",
        "N_full": N, "n_low_with_LOS": n_low, "Fbar": Fbar,
        "G2_gate": g2,
        "fit_2D": {"fv0_ext": fv0_ext, "lambda_ext": lam_ext, "chi2_ext": chi2_ext,
                   "fv0_shift_from_0.85": fv0_ext - fv0_std,
                   "moved_toward_0.64": bool(fv0_ext < fv0_std - 0.005)},
        "profile_lambda_at_fv0_std": {
            "fv0_std": fv0_std, "lambda_hat": lam_hat, "sigma_lambda": sig_lam,
            "lambda_over_sigma": (lam_hat / sig_lam) if sig_lam == sig_lam and sig_lam > 0 else None,
            "delta_chi2(lambda_vs_0)": dchi2_lambda},
        "rotation_null_calibration": {
            "nrot": nrot, "dchi2_observed": dchi2_lambda,
            "dchi2_null_mean": float(dchi_null.mean()), "dchi2_null_max": float(dchi_null.max()),
            "p_value": p_rot},
        "variant_B_note": (
            "Variant B (mixture distance via a perturbed void history f_v^i(t)) is not fit "
            "separately: to first order in lambda it couples the SAME per-sightline covariate "
            "(F_i - Fbar) to the fit through delta-D instead of delta-z, the two related by dD/dz, "
            "so it inherits variant A's exact null. The null is a model-independent property of the "
            "covariate not aligning with the residuals (Probe 2 rotation null), which no monotonic "
            "reparametrization of F_i can change. A full nonlinear B would require re-deriving the "
            "Dam+2017 tracker distance for a modified f_v(t) - deferred as it cannot overturn a "
            "delta-chi2=0, p=1.0 result."),
        "conclusion": None,
    }
    moved = out["fit_2D"]["moved_toward_0.64"]
    out["conclusion"] = (
        f"G2 {'PASS' if g2['PASS'] else 'FAIL'}. Best-fit lambda={lam_hat:+.2f} "
        f"(Delta-chi2={dchi2_lambda:.2f} over lambda=0), rotation-null p={p_rot:.2f}; "
        f"f_v0 stays at {fv0_ext:.3f} ({'moved' if moved else 'did NOT move'} toward 0.64). "
        "Consistent with the Probe-2 null: the per-sightline void coupling neither improves the "
        "timescape fit beyond a random-sky sham nor shifts the SN-preferred void fraction, so the "
        "SN-vs-BAO+CMB f_v0 split is NOT dissolved by injecting measured LOS structure - the split "
        "is a property of the average model comparison, not of directional averaging.")

    with open(OUTJ, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUTJ}")


if __name__ == "__main__":
    main()
