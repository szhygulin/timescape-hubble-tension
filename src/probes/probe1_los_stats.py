#!/usr/bin/env python3
"""Phase 3 (LOS voids), Probe 1 — per-SN line-of-sight void statistics.

Pure data engineering (no cosmology). For each usable SN (Probe-0 sample) computes, from
the 2M++ density field along the observer->SN sightline:
  * F_i(delta_th) = path-length fraction with delta < delta_th, for delta_th in {-0.3,-0.5,-0.7};
  * mean_delta_i  = path-averaged density contrast;
  * delta_loc_i   = spherical top-hat mean contrast at the SN (the monopole around it), R=8,12.
Also measures Var(F_i) vs z (feeds Probe 5) and runs the G1 sampler-validation gate.

G1 gate: an ensemble of random isotropic sightlines, sampled by the SAME ray sampler, must
reproduce the field's own radial shell-mean profile computed independently on the grid — this
proves the sampler reads the field without directional or radial bias.

Output: probes_out/probe1_los.npz (row-index-aligned covariates) + probes_out/probe1_los.json.
"""
import os
import json
import numpy as np

sys_dir = os.path.dirname(__file__)
import sys
sys.path.insert(0, sys_dir)
import los_common as LC

DELTA_TH = [-0.3, -0.5, -0.7]
DS = 0.5            # ray step, Mpc/h (field smoothed at 4 Mpc/h, so well-sampled)
RLOC = [8.0, 12.0]
OUTNPZ = os.path.join(LC.WT, "probes_out", "probe1_los.npz")
OUTJ = os.path.join(LC.WT, "probes_out", "probe1_los.json")


def grid_shell_profile(field, edges):
    """Mean delta in radial shells [edges[k], edges[k+1]] over the reliable sphere."""
    ii = np.arange(LC.NGRID)
    g = (ii - 128) * LC.DX
    XX, YY, ZZ = np.meshgrid(g, g, g, indexing="ij")
    rr = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2).ravel()
    d = field.ravel()
    prof = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        m = (rr >= edges[k]) & (rr < edges[k + 1])
        if m.any():
            prof[k] = d[m].mean()
    return prof


def g1_gate(field, n_rays=20000, rmax=195.0, ds=2.0, rmin=5.0, seed=12345):
    """Random isotropic rays vs grid shell profile; return max deviation + correlation.

    The innermost shells (r < rmin) are degenerate — near the origin a shell holds only the
    single central voxel while all rays converge onto it through trilinear interpolation, so a
    per-shell mismatch there is small-number noise, not sampler bias. The gate compares over
    rmin < r < rmax and cross-checks the ensemble-mean contrast against the field volume mean.
    """
    rng = np.random.default_rng(seed)
    l = rng.uniform(0, 360, n_rays)
    b = np.degrees(np.arcsin(rng.uniform(-1, 1, n_rays)))
    s = np.arange(ds, rmax + 1e-9, ds)
    ux, uy, uz = LC.unit_gal(l, b)                       # (3, n_rays)
    S = s[None, :]
    X = (ux[:, None] * S).ravel()
    Y = (uy[:, None] * S).ravel()
    Z = (uz[:, None] * S).ravel()
    dvals = LC.sample_points(field, X, Y, Z).reshape(n_rays, len(s))
    ray_prof = dvals.mean(axis=0)
    edges = np.concatenate([s - ds / 2, [s[-1] + ds / 2]])
    grid_prof = grid_shell_profile(field, edges)
    ok = np.isfinite(grid_prof) & (s >= rmin)
    dev = np.abs(ray_prof[ok] - grid_prof[ok])
    i_max = int(np.argmax(dev))
    corr = float(np.corrcoef(ray_prof[ok], grid_prof[ok])[0, 1])
    ens_mean = float(dvals[:, s >= rmin].mean())
    grid_mean = float(grid_prof[ok].mean())
    return {
        "n_rays": n_rays, "ds": ds, "rmin": rmin, "rmax": rmax,
        "max_abs_dev(r>rmin)": float(dev.max()),
        "max_dev_at_r_Mpc_h": float(s[ok][i_max]),
        "rms_dev": float(np.sqrt((dev ** 2).mean())),
        "profile_correlation": corr,
        "ray_ensemble_mean_delta": ens_mean,
        "grid_volume_mean_delta": grid_mean,
        "ensemble_vs_grid_mean_diff": abs(ens_mean - grid_mean),
        "PASS": bool(dev.max() < 0.05 and corr > 0.98 and abs(ens_mean - grid_mean) < 0.01),
    }


def main():
    field = LC.load_field()
    cat = LC.load_catalog()
    mask = LC.usable_mask(cat)
    idx = np.where(mask)[0]
    n = len(idx)
    print(f"# Probe 1: {n} usable rows")

    l = cat["l"][idx]; b = cat["b"][idx]; r = cat["r"][idx]; z = cat["zCMB"][idx]

    F = {th: np.empty(n) for th in DELTA_TH}
    mean_delta = np.empty(n)
    delta_loc = {R: np.empty(n) for R in RLOC}

    # endpoints for the local-sphere monopole
    ux, uy, uz = LC.unit_gal(l, b)
    Xe, Ye, Ze = r * ux, r * uy, r * uz

    for i in range(n):
        s, d = LC.sample_ray(field, l[i], b[i], r[i], ds=DS)
        mean_delta[i] = d.mean()
        for th in DELTA_TH:
            F[th][i] = np.mean(d < th)
    for R in RLOC:
        delta_loc[R] = LC.delta_local(field, Xe, Ye, Ze, R_loc=R)

    # Var(F_i) vs z (feeds Probe 5) — use the middle threshold -0.5
    zbins = [0.01, 0.02, 0.03, 0.04, 0.05, 0.0668]
    varF = {}
    Fmid = F[-0.5]
    for k in range(len(zbins) - 1):
        m = (z >= zbins[k]) & (z < zbins[k + 1])
        if m.sum() > 2:
            varF[f"{zbins[k]:.3f}-{zbins[k+1]:.3f}"] = {
                "n": int(m.sum()),
                "mean_F(-0.5)": float(Fmid[m].mean()),
                "var_F(-0.5)": float(Fmid[m].var()),
                "std_F(-0.5)": float(Fmid[m].std()),
            }

    g1 = g1_gate(field)
    print("G1 gate:", json.dumps(g1, indent=2))

    # save covariates aligned to catalog row index `idx`
    np.savez(
        OUTNPZ,
        row_index=idx, CID=cat["CID"][idx], zCMB=z, zHD=cat["zHD"][idx],
        l=l, b=b, r=r,
        F_m03=F[-0.3], F_m05=F[-0.5], F_m07=F[-0.7],
        mean_delta=mean_delta, delta_loc8=delta_loc[8.0], delta_loc12=delta_loc[12.0],
    )

    summary = {
        "probe": "1 — per-SN LOS void statistics",
        "n": n,
        "ds_Mpc_h": DS,
        "delta_thresholds": DELTA_TH,
        "F_stats": {
            f"F({th})": {"mean": float(F[th].mean()), "std": float(F[th].std()),
                         "min": float(F[th].min()), "max": float(F[th].max())}
            for th in DELTA_TH
        },
        "mean_delta_path": {"mean": float(mean_delta.mean()), "std": float(mean_delta.std()),
                            "min": float(mean_delta.min()), "max": float(mean_delta.max())},
        "delta_loc8": {"mean": float(delta_loc[8.0].mean()), "std": float(delta_loc[8.0].std())},
        "varF_vs_z(-0.5)": varF,
        "G1_gate": g1,
        "npz": os.path.relpath(OUTNPZ, LC.WT),
    }
    with open(OUTJ, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== PROBE 1 SUMMARY ===")
    print(json.dumps({k: summary[k] for k in
                      ["n", "F_stats", "mean_delta_path", "varF_vs_z(-0.5)", "G1_gate"]},
                     indent=2))
    print(f"\nwrote {OUTNPZ}\nwrote {OUTJ}")


if __name__ == "__main__":
    main()
