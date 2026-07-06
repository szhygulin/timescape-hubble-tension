# Phase 3 / Probe 0 — feasibility gate + sample definition (executed)

*Executed 2026-07-06 on the `significance-audit` branch. Gate for `PLAN_los_voids.md`.
Machinery: `src/probes/fetch_twompp.py`, `src/probes/probe0_sample.py`;
outputs `probes_out/probe0_sample.json` and the frozen sample
`probes_out/probe0_usable_sample.csv`.*

## Verdict: PASS — proceed to Probes 1–2, no footprint restriction needed.

An all-sky density field at z < 0.067 is obtainable, validated, and in hand; the usable
supernova sample (460 unique SNe / 573 catalog rows) is **more than double** the plan's
pre-registered pessimistic estimate (150–250). The plan's fallback branch (prespecify a
footprint-restricted subsample and its power loss) is **not triggered**.

## What was acquired

The reconstructed **2M++ density field** of Carrick, Turnbull, Lavaux & Hudson 2015
(MNRAS 450, 317; arXiv:1504.04627), from `cosmicflows.iap.fr/download`:

- `twompp_density.npy` — 257³ float64, Galactic Cartesian comoving, cell centres −200…+200
  Mpc/h (spacing 1.5625 Mpc/h), Local Group at voxel [128,128,128]. Value is δ_g\*, the
  luminosity-weighted galaxy density contrast, real space, Gaussian-smoothed at 4 Mpc/h.
- Reliable to ~200 Mpc/h ≈ z 0.067 — the field's own stated limit, and the sample's z cut.

Kept out of the repo (136 MB, regenerable) under `external_data/` (git-ignored); re-fetched
idempotently by `src/probes/fetch_twompp.py`.

## Validation (all pass)

**Field format.** shape (257,257,257) float64; global mean 0.0031 and mean-within-200-Mpc/h
0.0060 (≈0, as a density contrast must be); min = −1.0 exactly (the physical void floor
δ ≥ −1); LG voxel δ = 0.29 (mild local overdensity, matching the README).

**Coordinate transform** (hardcoded J2000-equatorial → Galactic rotation, no astropy
dependency; accurate far below the 0.8° grid cell at 100 Mpc/h): North Galactic Pole →
b = 89.9997°; Galactic Centre → (l,b) = (0,0) to 1e-5 deg.

**Sign & convention vs known structure** (δ_g\* sampled toward each): Virgo 8.3, Coma 10.0,
Perseus–Pisces 3.8, Great Attractor 1.1 — all overdense as they must be; Local Void −0.91
— underdense near the floor. The equatorial→Galactic→Cartesian chain and the sign of δ are
therefore both correct.

## Usable sample definition (frozen)

**Cut:** `IS_CALIBRATOR==0 AND zHD>0.01 AND comoving_dist(zCMB, flat ΛCDM Ωm=0.315) < 200 Mpc/h`.
zCMB (not zHD) places each SN in the box, per the plan's central caveat — zHD already carries
the 2M++ peculiar-velocity correction, so using it downstream would regress the data on
itself. The z>0.01 lower cut is the paper's own cosmology-sample definition.

| Quantity | Value |
|---|---|
| Cosmology sample (paper) | 1580 rows |
| **Usable rows (r < 200 Mpc/h)** | **573** |
| **Unique SNe (by CID)** | **460** |
| In Zone of Avoidance \|b\|<10° / <5° | 6 / 0 |
| Radial: r<125 / <150 / <180 / <200 | 481 / 530 / 559 / 573 |
| Redshift 0.01–0.02 / –0.03 / –0.04 / –0.05 / –0.067 | 149 / 197 / 121 / 55 / 43 |

Frozen to `probes_out/probe0_usable_sample.csv` (CID, zCMB, zHD, RA, DEC, l, b, r, endpoint δ),
sorted by comoving distance — Probes 1–5 consume exactly this set.

## Notes carried forward (not gate-blocking)

1. **Duplicates.** 573 rows = 460 unique SNe: Pantheon+ repeats the same SN across surveys,
   correlated in the full STAT+SYS covariance. Keep all rows for the GLS (matches the paper's
   machinery and the covariance); report 460 as the independent-sightline count.
2. **Endpoint (host) density is biased overdense** — median δ at SN positions = 1.06, only
   17% underdense. Expected: SNe trace galaxies trace mass. Probe 1's *path-integrated*
   covariates (F_i, path-averaged δ) largely wash this out, but Probe 2 must keep the
   host-density endpoint out of the covariate and control the residual selection.
3. **ZoA is a non-issue** (6 rows), so no |b| mask is needed; note it as a caveat only.
4. **Possible low-z extension:** 44 non-calibrator SNe sit at 0 < zHD ≤ 0.01 (peculiar-velocity
   dominated). Excluded by the paper's cut; revisit only if Probe 2 wants the deepest-variance
   regime and can model the PV floor.
5. **Pre-fit TODO before Probe 2** (unchanged from the plan): write down the pre-registered
   sign of the regression slope λ *before* fitting (more LOS void ⇒ faster local expansion ⇒
   at fixed z the SN is nearer/brighter ⇒ fixed residual-convention sign), plus the negative
   (rotated-sky) and positive (zHD-swap) controls.

## Recommended next step

Probe 1 (per-SN LOS void statistics: F_i, path-averaged δ, shell δ — pure data engineering,
G1 sightline-vs-global gate) → Probe 2 (model-independent GLS regression of Hubble residuals
on F_i, the cheap decisive test; pre-registered sign + permutation p-value + rotated-sky and
zHD controls). Probe 2's outcome dictates whether Probe 3 (the one-parameter per-sightline
timescape extension) is warranted.
