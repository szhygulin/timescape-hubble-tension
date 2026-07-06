#!/usr/bin/env python3
"""Phase 3 (LOS voids), Probe 5 — analytic ceiling for the per-sightline effect vs z.

Shows WHY all discriminating power from line-of-sight structure sits at z <~ 0.1, independent
of the coupling strength. Independent-cell counting gives Var(F_i) ~ L_void / D(z) (more
independent voids average out along a longer beam), so the per-sightline magnitude modulation
from the Probe-3A coupling declines with redshift while the Pantheon+ per-SN error budget
(intrinsic + lensing sigma_lens ~ 0.055 z) does not. The measured Var(F) at z<0.067 (Probe 1)
calibrates the amplitude; the analytic 1/sqrt(D) form extrapolates it to z=0.3.

Arithmetic only - no fitting of cosmology. Output: probes_out/probe5_ceiling.json.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import los_common as LC

C_KM = LC.C_KM
OUTJ = os.path.join(LC.WT, "probes_out", "probe5_ceiling.json")


def dimless_D(z, om=LC.OM_FID):
    """Dimensionless comoving distance H0 D_C / c = int_0^z dz/E."""
    return LC.comoving_dist_mpc_h(z) / (C_KM / LC.H100)


def mu_fid(z, om=LC.OM_FID):
    """Fiducial distance modulus up to an additive constant (H0/M absorbed)."""
    return 5.0 * np.log10((1 + z) * np.maximum(dimless_D(z), 1e-12))


def main():
    # measured std(F) vs z from Probe 1
    d1 = np.load(os.path.join(LC.WT, "probes_out", "probe1_los.npz"))
    z = d1["zCMB"]; F = d1["F_m05"]
    edges = np.array([0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.0668])
    zc, stdF, nbin = [], [], []
    for k in range(len(edges) - 1):
        m = (z >= edges[k]) & (z < edges[k + 1])
        if m.sum() >= 8:
            zc.append(float(z[m].mean())); stdF.append(float(F[m].std())); nbin.append(int(m.sum()))
    zc = np.array(zc); stdF = np.array(stdF)
    Dc_mpc = dimless_D(zc) * (C_KM / LC.H100)      # beam length in Mpc/h
    fbar = float(F.mean()); f1f = fbar * (1 - fbar)

    # independent-cell counting: Var(F) = f(1-f) * L_void / D  =>  L_void = Var(F)*D/[f(1-f)].
    # Report the implied L_void per bin. It is NOT constant (larger at higher z): at very low z the
    # short beam holds only ~1-2 coherence lengths so Var(F) saturates below the many-cell scaling,
    # hence the empirical decline is shallower than the asymptotic D^-1. The higher-z bins, in the
    # many-cell regime, give the applicable L_void for extrapolation.
    L_implied = (stdF ** 2 * Dc_mpc / max(f1f, 1e-6))
    L_hi = float(np.median(L_implied[zc > 0.03]))   # calibrated where the many-cell model applies

    def stdF_model(zq):
        """Independent-cell UPPER bound: sqrt(f(1-f) L_void / D), capped at the 1-cell limit."""
        D_mpc = dimless_D(zq) * (C_KM / LC.H100)
        v = f1f * L_hi / D_mpc
        return np.sqrt(np.minimum(v, f1f))

    # magnitude modulation from the Probe-3A coupling: dmu = (dmu/dz)*(1+z)*lambda*(F-Fbar)*D.
    # RMS over sightlines = (dmu/dz)*(1+z)*lambda*stdF(z)*D(z). The DATA-CONSTRAINED coupling is
    # |lambda| < 0.05 (Probe 3: sigma_lambda=0.017, lambda_hat=0); larger values are illustrative
    # only (lambda=0.5 is ~30 sigma excluded) and show the geometric z-shape, not a real amplitude.
    zq = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30])
    dz = 1e-4
    dmudz = (mu_fid(zq + dz) - mu_fid(zq - dz)) / (2 * dz)
    Dq = dimless_D(zq)
    # data-constrained coupling ceiling: 3 sigma_lambda from Probe 3, INFLATED by the rotation-null
    # width (Probe 3's null dchi2 mean > 1 shows the naive delta-chi2=1 error understates the
    # estimator scatter, consistent with the Probe-2 lesson).
    p3 = json.load(open(os.path.join(LC.WT, "probes_out", "probe3_los_timescape.json")))
    sig_lam3 = p3["profile_lambda_at_fv0_std"]["sigma_lambda"]
    infl = float(np.sqrt(max(p3["rotation_null_calibration"]["dchi2_null_mean"], 1.0)))
    LAM_CEIL = round(3.0 * sig_lam3 * infl, 3)
    rms_dmu = {}
    for lam in [LAM_CEIL, 0.5, 1.0]:
        tag = f"lambda={lam}" + (" (data-constrained 3sigma ceiling)" if lam == LAM_CEIL else " (illustrative, excluded)")
        rms_dmu[tag] = (dmudz * (1 + zq) * lam * stdF_model(zq) * Dq).tolist()

    # Pantheon+ per-SN error budget vs z (median diagonal magnitude error from the full catalog)
    cat = LC.load_catalog()
    cosmo = (cat["IS_CALIBRATOR"] == 0) & (cat["zHD"] > 0.01)
    zz = cat["zHD"][cosmo]; sig = cat["m_b_corr_err_DIAG"][cosmo]
    budget = []
    for zt in zq:
        m = (zz >= zt * 0.8) & (zz < zt * 1.25)
        budget.append(float(np.median(sig[m])) if m.sum() >= 5 else None)
    sig_lens = (0.055 * zq).tolist()

    # ratio of the data-constrained per-sightline modulation to the error budget vs z
    bud = np.array([b if b is not None else np.nan for b in budget])
    ceil = np.array(rms_dmu[[k for k in rms_dmu if k.startswith(f"lambda={LAM_CEIL}")][0]])
    ratio_ceiling = (ceil / bud).tolist()

    out = {
        "probe": "5 — analytic z>0.1 ceiling",
        "measured_stdF_vs_z": [{"z": float(a), "stdF(-0.5)": float(b), "n": int(c),
                                "implied_L_void_Mpc_h": float(d)}
                               for a, b, c, d in zip(zc, stdF, nbin, L_implied)],
        "independent_cell_model": {
            "f(1-f)": f1f,
            "L_void_range_Mpc_h": [float(L_implied.min()), float(L_implied.max())],
            "L_void_calibrated(z>0.03)_Mpc_h": L_hi,
            "note": "implied L_void rises from ~low-z (saturated, few-cell) to the many-cell value; "
                    "the extrapolation uses the many-cell L_void as an UPPER bound on std(F)."},
        "z_grid": zq.tolist(),
        "dmu_dz": dmudz.tolist(),
        "RMS_dmu_vs_z(mag)": rms_dmu,
        "Pantheon_error_budget_median(mag)": budget,
        "sigma_lens(0.055z)": sig_lens,
        "ratio_dataconstrained_ceiling_to_budget": ratio_ceiling,
        "lambda_ceiling(3sigma, rotation-null-inflated)": LAM_CEIL,
        "conclusion": (
            f"Two independent reasons the LOS effect is negligible and low-z-weighted. (1) Amplitude: "
            f"the data cap the coupling at |lambda|<{LAM_CEIL:.3f} (Probe 3, 3 sigma, rotation-null "
            f"inflated), so the per-sightline "
            f"magnitude modulation is at most ~{max(ceil):.3f} mag - a few percent of the ~0.2 mag "
            f"Pantheon+ per-SN error at every redshift (ratio {min(ratio_ceiling):.02f}-{max(ratio_ceiling):.02f}). "
            f"(2) Geometry: Var(F) ~ f(1-f) L_void/D(z) with L_void~{L_hi:.0f} Mpc/h (many-cell regime), "
            "so even a hypothetical order-unity coupling declines with beam length and is concentrated "
            "at z<~0.1. Both independently explain WHY the published timescape SN preference is "
            "low-z-driven - a low-z/covariance effect, not a high-z cosmological signal.")}

    with open(OUTJ, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUTJ}")


if __name__ == "__main__":
    main()
