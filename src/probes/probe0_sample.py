#!/usr/bin/env python3
"""Phase 3 (LOS voids), Probe 0 — feasibility gate + SN sample definition.

Establishes whether the line-of-sight void programme (PLAN_los_voids.md) is executable
on the in-repo Pantheon+ sample against the external 2M++/Carrick+2015 density field, and
writes the prespecified usable-SN sample definition that gates every later probe.

What it does (no cosmology fitting — pure data engineering):
  1. Load the 2M++ density field (external_data/twompp_density.npy) and self-check its
     format (shape, global mean, Local-Group voxel).
  2. Self-validate the hardcoded J2000-equatorial -> Galactic rotation against the North
     Galactic Pole and Galactic-Centre directions, then sanity-check the coordinate
     convention against known nearby overdensities (Virgo, Coma, Great Attractor) and the
     Local Void.
  3. Load the Pantheon+ cosmology sample (IS_CALIBRATOR==0, zHD>0.01 — the paper's cut),
     map each SN (RA, DEC, zCMB) into the box, and count how many land inside the
     reconstruction's reliable radius (r < 200 Mpc/h ~ z 0.067), by radial shell and
     with the Zone-of-Avoidance flagged.
  4. Report the endpoint density-contrast statistics at the SN positions (a sampling
     smoke test), and write probes_out/probe0_sample.json with the sample definition.

CRITICAL CAVEAT (baked in): zHD already contains 2M++ peculiar-velocity corrections, so a
later covariate regression must use zCMB, never zHD (else it regresses the data on itself).
Here zCMB is used only to *place* each SN in the box; the redshift cut zHD>0.01 follows the
paper's cosmology-sample definition.
"""
import os
import sys
import json
import numpy as np
from scipy.integrate import cumulative_trapezoid

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIELD = os.path.join(WT, "external_data", "twompp_density.npy")
DATA = os.path.join(WT, "src", "data", "PantheonSH0ES.dat")
OUTJ = os.path.join(WT, "probes_out", "probe0_sample.json")

C_KM = 299792.458          # speed of light, km/s
OM_FID = 0.315             # fiducial flat-LCDM matter density for the z->distance map
H100 = 100.0               # distances are in Mpc/h, so H0 = 100 h km/s/Mpc
NGRID = 257
LMIN, LMAX = -200.0, 200.0
DX = (LMAX - LMIN) / (NGRID - 1)   # 1.5625 Mpc/h
R_RELIABLE = 200.0         # Mpc/h; 2M++ reconstruction limit (README) ~ z 0.067

# J2000 (FK5/ICRS-approx) equatorial -> Galactic rotation matrix (Hipparcos convention):
# r_gal = R @ r_eq, with r_eq = (cosDec cosRA, cosDec sinRA, sinDec).
R_EQ2GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [+0.4941094279, -0.4448296300, +0.7469822445],
    [-0.8676661490, -0.1980763734, +0.4559837762],
])


def eq_to_gal_lb(ra_deg, dec_deg):
    """RA/DEC (J2000, degrees) -> Galactic (l, b) in degrees."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    r_eq = np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])
    xg, yg, zg = R_EQ2GAL @ r_eq
    l = np.degrees(np.arctan2(yg, xg)) % 360.0
    b = np.degrees(np.arcsin(np.clip(zg, -1, 1)))
    return l, b


def gal_lbr_to_xyz(l_deg, b_deg, r):
    """Galactic (l, b, r) -> Galactic Cartesian (X, Y, Z), same units as r."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    return (r * np.cos(b) * np.cos(l),
            r * np.cos(b) * np.sin(l),
            r * np.sin(b))


def comoving_dist_mpc_h(z, om=OM_FID, n=200000):
    """Fiducial flat-LCDM comoving distance in Mpc/h for a redshift array."""
    zg = np.linspace(0.0, float(np.max(z)) * 1.0001 + 1e-6, n)
    invE = 1.0 / np.sqrt(om * (1 + zg) ** 3 + (1 - om))
    chi = (C_KM / H100) * cumulative_trapezoid(invE, zg, initial=0.0)
    return np.interp(z, zg, chi)


def sample_field(field, X, Y, Z):
    """Nearest-cell density contrast at Galactic Cartesian (X, Y, Z) in Mpc/h.

    field[i, j, k] with X=(i-128)*DX etc.; NaN if outside the cube.
    """
    i = np.rint((X - LMIN) / DX).astype(int)
    j = np.rint((Y - LMIN) / DX).astype(int)
    k = np.rint((Z - LMIN) / DX).astype(int)
    inside = ((i >= 0) & (i < NGRID) & (j >= 0) & (j < NGRID) & (k >= 0) & (k < NGRID))
    out = np.full(X.shape, np.nan)
    ii, jj, kk = i[inside], j[inside], k[inside]
    out[inside] = field[ii, jj, kk]
    return out


# ---------------------------------------------------------------------------
# 1. Load & self-check the density field
# ---------------------------------------------------------------------------
def load_field():
    d = np.load(FIELD)
    checks = {
        "shape": list(d.shape),
        "dtype": str(d.dtype),
        "global_mean_full_cube": float(d.mean()),
        "LG_voxel_[128,128,128]": float(d[128, 128, 128]),
        "min": float(d.min()),
        "max": float(d.max()),
    }
    # mean over the reliable sphere (should be ~0 for a density contrast)
    ii = np.arange(NGRID)
    Xg = (ii - 128) * DX
    XX, YY, ZZ = np.meshgrid(Xg, Xg, Xg, indexing="ij")
    rr = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
    checks["mean_within_100Mpc"] = float(d[rr < 100].mean())
    checks["mean_within_200Mpc"] = float(d[rr < 200].mean())
    return d, checks


def coord_selfcheck():
    """Verify the rotation matrix on NGP / Galactic Centre and known structures."""
    out = {}
    # NGP: RA=192.85948, Dec=+27.12825 -> b ~ +90
    l, b = eq_to_gal_lb(192.85948, 27.12825)
    out["NGP_should_be_b90"] = {"l": float(l), "b": float(b)}
    # Galactic Centre: RA=266.40499, Dec=-28.93617 -> (l,b) ~ (0,0)
    l, b = eq_to_gal_lb(266.40499, -28.93617)
    out["GC_should_be_l0_b0"] = {"l": float(l), "b": float(b)}
    return out


def structure_selfcheck(d):
    """Sample the field toward known overdensities/voids to fix the sign & convention.

    Positions given in Galactic (l, b) with an approximate distance in Mpc/h.
    """
    targets = {
        "Virgo_cluster":   (283.8, 74.5, 12.0),   # overdense (nearby)
        "Coma_cluster":    (58.1, 87.9, 69.0),    # overdense
        "Great_Attractor": (307.0, 9.0, 45.0),    # overdense (near ZoA)
        "Perseus_Pisces":  (140.0, -22.0, 50.0),  # overdense
        "Local_Void":      (60.0, -15.0, 25.0),   # underdense
    }
    res = {}
    for name, (l, b, r) in targets.items():
        X, Y, Z = gal_lbr_to_xyz(np.array([l]), np.array([b]), np.array([r]))
        res[name] = {"lbr": [l, b, r], "delta_g": float(sample_field(d, X, Y, Z)[0])}
    return res


# ---------------------------------------------------------------------------
# 2. Load the Pantheon+ cosmology sample
# ---------------------------------------------------------------------------
def load_sne():
    with open(DATA) as f:
        header = f.readline().split()
        idx = {n: i for i, n in enumerate(header)}
        rows = [ln.split() for ln in f]
    col = lambda n: np.array([float(r[idx[n]]) for r in rows])
    iscal = np.array([int(float(r[idx["IS_CALIBRATOR"]])) for r in rows])
    d = {n: col(n) for n in ["zHD", "zCMB", "zHEL", "RA", "DEC", "VPEC", "VPECERR"]}
    d["IS_CALIBRATOR"] = iscal
    d["CID"] = np.array([r[idx["CID"]] for r in rows])
    return d


def main():
    print("# Phase 3 / Probe 0 — feasibility gate + sample definition\n")

    d, field_checks = load_field()
    print("Density field:", json.dumps(field_checks, indent=2))
    coord = coord_selfcheck()
    print("\nCoordinate self-check:", json.dumps(coord, indent=2))
    struct = structure_selfcheck(d)
    print("\nStructure self-check (delta_g toward known objects):",
          json.dumps(struct, indent=2))

    sn = load_sne()
    cosmo = (sn["IS_CALIBRATOR"] == 0) & (sn["zHD"] > 0.01)
    n_cosmo = int(cosmo.sum())

    # place the full cosmology sample in the box using zCMB
    z = sn["zCMB"]
    r = comoving_dist_mpc_h(np.maximum(z, 0.0))
    l, b = eq_to_gal_lb(sn["RA"], sn["DEC"])
    X, Y, Z = gal_lbr_to_xyz(l, b, r)
    delta_end = sample_field(d, X, Y, Z)

    # radial shells within the cosmology sample
    def count(mask):
        return int((cosmo & mask).sum())

    shells = {
        "r<125 (2MRS-limited depth, all-sky reliable)": count(r < 125),
        "r<150": count(r < 150),
        "r<180": count(r < 180),
        "r<200 (reconstruction limit ~ z 0.067)": count(r < R_RELIABLE),
    }

    # the headline usable sample: cosmology cut AND inside the reliable sphere
    usable = cosmo & (r < R_RELIABLE) & np.isfinite(delta_end)
    n_usable = int(usable.sum())

    # Zone-of-Avoidance breakdown within the usable sample
    zoa5 = usable & (np.abs(b) < 5)
    zoa10 = usable & (np.abs(b) < 10)

    # endpoint density-contrast stats over the usable sample (sampling smoke test + G1-lite)
    du = delta_end[usable]
    endpoint = {
        "n": n_usable,
        "mean_delta_g": float(np.mean(du)),
        "median_delta_g": float(np.median(du)),
        "std_delta_g": float(np.std(du)),
        "min": float(np.min(du)),
        "max": float(np.max(du)),
        "frac_underdense(delta<0)": float(np.mean(du < 0)),
    }

    # redshift structure of the usable sample
    zu = z[usable]
    zbins = {
        "0.01-0.02": int(((zu >= 0.01) & (zu < 0.02)).sum()),
        "0.02-0.03": int(((zu >= 0.02) & (zu < 0.03)).sum()),
        "0.03-0.04": int(((zu >= 0.03) & (zu < 0.04)).sum()),
        "0.04-0.05": int(((zu >= 0.04) & (zu < 0.05)).sum()),
        "0.05-0.067": int(((zu >= 0.05) & (zu < 0.0668)).sum()),
    }

    # how many non-calibrator SNe sit below the paper's z>0.01 cut (potential extension)
    below_cut = int(((sn["IS_CALIBRATOR"] == 0) & (sn["zHD"] <= 0.01) & (sn["zHD"] > 0)).sum())

    result = {
        "probe": "0 — feasibility gate + sample definition",
        "field_source": "Carrick+2015 2M++ (cosmicflows.iap.fr/download), 257^3, +-200 Mpc/h",
        "field_checks": field_checks,
        "coord_selfcheck": coord,
        "structure_selfcheck": struct,
        "n_cosmology_sample(iscal==0, zHD>0.01)": n_cosmo,
        "radial_shells_within_cosmology_sample": shells,
        "usable_sample_definition": {
            "cut": "IS_CALIBRATOR==0 AND zHD>0.01 AND comoving_dist(zCMB, LCDM Om=0.315) < 200 Mpc/h",
            "n_usable_rows": n_usable,
            "n_unique_SNe(by CID)": int(len(set(sn["CID"][usable]))),
            "note": "Pantheon+ has duplicate rows (same SN, multiple surveys) correlated in "
                    "the full covariance; GLS uses all rows, so the row count matches the "
                    "paper's machinery while unique sightlines is the independent-info count",
            "n_in_ZoA_|b|<5": int(zoa5.sum()),
            "n_in_ZoA_|b|<10": int(zoa10.sum()),
            "n_outside_ZoA_|b|>=10": int((usable & (np.abs(b) >= 10)).sum()),
        },
        "usable_redshift_bins": zbins,
        "endpoint_density_stats": endpoint,
        "n_noncalibrator_below_zHD_0.01(possible_extension)": below_cut,
    }

    # freeze the exact prespecified sample so later probes consume this set verbatim
    OUTCSV = os.path.join(WT, "probes_out", "probe0_usable_sample.csv")
    order = np.where(usable)[0]
    order = order[np.argsort(r[order])]   # sorted by comoving distance
    with open(OUTCSV, "w") as f:
        f.write("CID,zCMB,zHD,RA_deg,DEC_deg,l_gal_deg,b_gal_deg,r_Mpc_h,delta_endpoint\n")
        for m in order:
            f.write(f"{sn['CID'][m]},{z[m]:.6f},{sn['zHD'][m]:.6f},"
                    f"{sn['RA'][m]:.5f},{sn['DEC'][m]:.5f},{l[m]:.4f},{b[m]:.4f},"
                    f"{r[m]:.3f},{delta_end[m]:.5f}\n")
    result["frozen_sample_csv"] = os.path.relpath(OUTCSV, WT)

    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    with open(OUTJ, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== SAMPLE DEFINITION ===")
    print(json.dumps({k: result[k] for k in [
        "n_cosmology_sample(iscal==0, zHD>0.01)",
        "radial_shells_within_cosmology_sample",
        "usable_sample_definition",
        "usable_redshift_bins",
        "endpoint_density_stats",
        "n_noncalibrator_below_zHD_0.01(possible_extension)",
    ]}, indent=2))
    print(f"\nwrote {OUTJ}")


if __name__ == "__main__":
    main()
