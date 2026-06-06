#!/usr/bin/env python3
"""
Szekeres inhomogeneous cosmology vs SN + BAO + CMB on the shared harness.

VERDICT: tractable = INTRACTABLE for a validated, fitting-relevant Szekeres
distance code in this harness. Observable distances in a *general* (non-LTB,
non-spherical) Szekeres metric have NO closed form and NO reduction to a 1-D
radial quadrature: d_L(z) requires numerically integrating the full set of null
geodesic ODEs in the inhomogeneous metric, PLUS the area/focusing (Sachs)
equation -- partial derivatives of the null-vector components along the ray --
from the *observer's* worldline outward, and the result is direction-dependent
(anisotropic) because Szekeres breaks all continuous symmetry. The fit space is
~8 parameters (6 metric-function coefficients + observer radial position r0 + Lambda),
none of which map onto the harness's single-quadrature Dc(z)/DM/DH/DV interface.

The harness's sn_chi2(Dc) and bao_cmb_chi2(predict) assume an isotropic,
observer-independent radial distance; a faithful Szekeres prediction would need
a ray-by-ray geodesic+Sachs solve per data point per direction, which cannot be
validated here against any published Szekeres d_L(z) curve -- because no such
fitted curve exists in the primary literature (see refs below). Implementing
unvalidated geodesic ODEs and fitting them would be fabrication, so we DO NOT.

This file therefore (a) documents the literature verdict and (b) runs the LCDM
reference through the harness to confirm the harness wiring is intact (the only
number we CAN validate here).

PRIMARY REFERENCES
------------------
* Meures & Bruni 2011, "Luminosity distance and redshift in the Szekeres
  inhomogeneous cosmological models" (arXiv:1005.2989): the distance method --
  affine null geodesic equations reduced to first-order ODEs integrated
  numerically to get the partial derivatives of the null-vector components
  (i.e. the area/Jacobi distance); applies to general Szekeres "without
  symmetries"; NO closed form.
* Bolejko & Celerier 2010, "Szekeres Swiss-Cheese model and supernova
  observations" (arXiv:1005.2584, PRD 82 103510): small-scale (~50 Mpc)
  Szekeres/LTB inhomogeneities CANNOT explain SN dimming without Lambda;
  structures >= 500 Mpc are required to fit SN data.
* "A ready-to-fit ... axially symmetric Szekeres spacetime" (arXiv:2604.16160,
  2026): even the *reduced* axially symmetric case is presented as a FRAMEWORK
  with ~8 free constants and the numerical fitting code "the subject of future
  work" -- NO joint SN+BAO+CMB fit, NO best-fit chi2/H0, NO published d_L(z).
* Camarena, Marra, Sakr & Clarkson 2022, "A void in the Hubble tension? The end
  of the line for the Hubble bubble" (CQG 39 184001, arXiv:2205.05422): the
  decisive observational verdict on the radial-inhomogeneity (LTB/LambdaLTB)
  family -- jointly fit to CMB+BAO+SN+H0+kSZ the data prefer only a SHALLOW void
  (delta_L ~ -0.04, r_out ~ 300 Mpc); the kSZ effect caps the void depth;
  "the H0 tension is not solved and the support for the LambdaLTB model
  vanishes" (residual ~3.2 sigma). Szekeres generalises LTB, so this kSZ-driven
  exclusion of deep local inhomogeneity applies a fortiori.

ESTABLISHED VERDICT: Szekeres (like LTB) is a viable description of *local*
small-amplitude inhomogeneity but does NOT resolve the 67-vs-73 H0 tension.
A void/inhomogeneity deep enough to mimic high local H0 is excluded by kSZ and
by the SN+BAO+CMB combination; the surviving inhomogeneity matches the standard
model's expectation. The mechanism is a late-time geometric distortion of
distances, which does NOT rescale the early-time sound horizon r_d that anchors
the BAO+CMB inverse distance ladder -- so it cannot shift the inferred H0 the
way a genuine early-universe modification would.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import harness as H

zHD, zHEL, mb, Cf = H.load_sn()

# --- LCDM reference through the harness (the one validatable number) ---
def joint_lcdm(Om):
    return H.sn_chi2(H.lcdm_Dc(zHD, Om)) + H.bao_cmb_chi2(H.lcdm_predict(Om))[0]

if __name__ == "__main__":
    r = minimize_scalar(joint_lcdm, bounds=(0.15, 0.45), method="bounded")
    Om = r.x
    chi_joint = r.fun
    _, a = H.bao_cmb_chi2(H.lcdm_predict(Om))
    H0 = H.H0_from_alpha(a)
    # SN-only and BAO+CMB-only LCDM for context
    sn_only = minimize_scalar(lambda Om: H.sn_chi2(H.lcdm_Dc(zHD, Om)),
                              bounds=(0.05, 0.6), method="bounded")
    bc_only = minimize_scalar(lambda Om: H.bao_cmb_chi2(H.lcdm_predict(Om))[0],
                              bounds=(0.15, 0.45), method="bounded")
    bao_only = minimize_scalar(lambda Om: H.bao_only_chi2(H.lcdm_predict(Om))[0],
                               bounds=(0.15, 0.45), method="bounded")

    print("=" * 72)
    print("SZEKERES inhomogeneous cosmology -- tractability assessment")
    print("=" * 72)
    print("Observable d_L(z) in a general Szekeres metric has NO closed form:")
    print("  requires per-ray null-geodesic + Sachs/area ODE integration from")
    print("  the observer outward; direction-dependent (anisotropic); ~8 params.")
    print("  No published fitted Szekeres d_L(z) curve exists to validate against.")
    print("  -> tractable = INTRACTABLE in this isotropic-Dc harness; not fabricating.")
    print("-" * 72)
    print("LCDM reference (harness validation -- the validatable number):")
    print(f"  joint   Om={Om:.3f}  chi2={chi_joint:.1f}  H0={H0:.1f}")
    print(f"          (expect Om~0.305, chi2~1402.2, H0~68.6)")
    print(f"  SN-only      Om={sn_only.x:.3f}  chi2={sn_only.fun:.1f}")
    print(f"  BAO-only     Om={bao_only.x:.3f}  chi2={bao_only.fun:.1f}")
    print(f"  BAO+CMB-only Om={bc_only.x:.3f}  chi2={bc_only.fun:.1f}")
    print("-" * 72)
    print("LITERATURE VERDICT (Camarena+2022, Bolejko&Celerier 2010):")
    print("  Szekeres/LTB local inhomogeneity does NOT resolve the H0 tension.")
    print("  kSZ caps void depth; deep voids excluded; data prefer a shallow")
    print("  void (delta~-0.04, r~300 Mpc); H0 tension remains ~3.2 sigma.")
    print("  Late-time distance distortion does not rescale early-time r_d.")
    print("=" * 72)
