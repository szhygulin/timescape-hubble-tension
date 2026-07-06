# Phase 3 — line-of-sight voids vs the single-parameter timescape average (executed)

*Executed 2026-07-06 on the `significance-audit` branch. Implements `PLAN_los_voids.md` in full:
Probes 0–5 + the T1 rider. Scripts `src/probes/{fetch_twompp,probe0_sample,probe1_los_stats,
probe2_gls,probe3_los_timescape,probe4_voidfrac,probe5_ceiling,probeT1_freeshape}.py` and
`src/probes/los_common.py`; machine-readable outputs in `probes_out/probe*.json`. The external
2M++/Carrick+2015 density field lives git-ignored in `external_data/` (fetched by
`fetch_twompp.py`).*

## The question

Timescape compresses all inhomogeneity into one global scalar `f_v0` on a one-parameter tracker,
and applies the same average distance–redshift curve to every sightline. The Pantheon+ SN fit
prefers `f_v0 ≈ 0.85`; the BAO+CMB fit prefers `≈ 0.64`. Three distinct senses of "the model is
too simple", each tested here:

- **T1 (temporal rigidity):** could *any* smooth void-history fit SN and BAO+CMB together?
- **T2 (directional averaging):** every SN gets the average curve, but the actual LOS void content
  varies SN-to-SN, strongly at z ≲ 0.1. Does injecting it reveal missed signal or dissolve the split?
- **T3 (external consistency):** is `f_v0 ≈ 0.85` even an allowed void volume fraction?

## Headline result

**T2 is rejected as the cause; T1 is the culprit; T3 disfavours the SN corner.** Injecting the
*measured* per-sightline void content changes nothing — it neither carries distance information
beyond the average model (Probe 2 null) nor moves the SN-preferred `f_v0` toward 0.64 (Probe 3
null), and the maximal possible per-sightline effect is a ≤0.05-mag, low-z-confined ceiling
(Probe 5). Meanwhile a *free smooth expansion history* reconciles SN+BAO+CMB as well as ΛCDM and
far better than the timescape tracker (T1 rider), and the SN-preferred `f_v0 ≈ 0.85` exceeds even
the most permissive observed void volume fraction (Probe 4). So the SN-vs-BAO+CMB split is the
**rigidity of the single-parameter tracker history**, not a directional-averaging artifact — and
the SN preference for deep voids is a low-z/covariance effect, not a physical void fraction. This
sharpens the paper's "recasting, not resolution" conclusion.

## Probe-by-probe

**Probe 0 — feasibility gate (PASS).** Acquired the all-sky 2M++ density field (257³, ±200 Mpc/h
≈ z 0.067). Coordinate transform and δ sign validated against NGP/Galactic-Centre and Virgo/Coma/
Great-Attractor (overdense) / Local Void (underdense). Usable sample **573 rows / 460 unique SNe**
at 0.01 < z < 0.067, all-sky, Zone-of-Avoidance negligible (6 rows) — 2× the plan's estimate, so no
footprint restriction. Frozen to `probes_out/probe0_usable_sample.csv`. (`REPORT_los_probe0.md`.)

**Probe 1 — per-SN LOS statistics.** Void fraction F_i(δ_th) at δ_th ∈ {−0.3,−0.5,−0.7}, path-
averaged δ, and local monopole δ_loc for all 573 sightlines. Sightlines are ~22% void (δ<−0.5) on
average; Var(F) is largest at z<0.03 and shrinks with beam length. **G1 sampler gate PASS**: 20000
random isotropic rays reproduce the field's radial shell profile (max deviation 0.019 at r=74
Mpc/h, correlation 0.999, ensemble-vs-grid mean agree to 0.0016).

**Probe 2 — model-independent GLS regression → NULL.** The decisive cheap test. Regressing SN
Hubble residuals `r(zCMB)` on the LOS covariates under the full STAT+SYS covariance, calibrated
against the authoritative **sky-rotation null**: mean_delta −0.9σ (p=0.40, seed-robust), F_void
−0.6σ (p=0.62). Two independent reasons the naïve signal is not real: the naïve slope is also
**wrong-sign** relative to the pre-registered prediction (the pre-registration's decision rule fails
on the sign clause alone), and its **σ_λ is understated ~2.7×** by the naïve GLS/permutation error
(which ignores the covariate's spatial coherence and the covariance's coherent systematic modes) —
a single rotated-sky sham reaches 3σ under the too-narrow null, so the apparent −2.1σ is
sub-threshold and not robust to a coherence-preserving null. G4 (`r` on zHD unchanged) and
`corr(mean_delta, VPEC) ≈ 0.01` independently show the covariate carries no peculiar-velocity
information. → The measured low-z structure adds nothing at Pantheon+ precision; the
single-parameter *average* is not the bottleneck.

**Probe 3A — per-sightline timescape extension → f̂_v0 does NOT move.** Fitting the expansion-
variance coupling `z_i → z_i + (1+z)λ(F_i−F̄)d_i` on the full 1580-SN timescape model. **G2 gate
PASS**: λ=0 reproduces the committed fit (f_v0=0.853, χ²=1391.545). Best-fit **λ̂=−0.012, Δχ²=0.63**
(~0.8σ for one parameter); against the rotation null the real sightlines give Δχ²=0.63 while random
skies average 1.50 (max 9.3), **p=0.55** — measured structure helps the fit no more than a random
direction. `f̂_v0` stays at 0.85 (a fine joint refit shifts it by −0.001, not toward 0.64). Variant
B inherits the null by first-order δz↔δD equivalence on the same covariate. → The split is not
dissolved by injecting measured LOS structure.

**Probe 4 — external void-fraction bracket → 0.85 disfavoured, 0.64 plausible.** Zero-fit T3 test.
Under the watershed / below-mean-density definition (the one mapping onto timescape's "regions
expanding faster than the mean", used by Williams et al. 2024 for the timescape comparison), the
observed present-day void volume fraction brackets **[0.50, 0.62]** (Williams+2024 numerical-
relativity 50–61.5%; Pan+2012 SDSS DR7 62%); timescape's own Planck fits cluster 0.627–0.695.
`f_v0 ≈ 0.64` sits at the top edge and matches these → plausible. `f_v0 ≈ 0.85` exceeds even the
maximum watershed filling → disfavoured under any structure-based definition. (Definition-dependence
stated explicitly; the load-bearing point is that 0.85 fails the *most permissive* definition.)

**Probe 5 — analytic z-ceiling → the effect is ≤0.04 mag and low-z-weighted.** Two independent
reasons the LOS effect is negligible: (1) *amplitude* — the data cap the coupling at |λ|<0.058
(Probe 3, 3σ, rotation-null-inflated), so RMS δμ ≤ 0.046 mag, only 6–15% of the ~0.2 mag per-SN
error, peaking at low z;
(2) *geometry* — Var(F) ~ f(1−f)·L_void/D(z) with L_void ~ 22 Mpc/h, so even a hypothetical
order-unity coupling declines with beam length. Independently explains WHY the published timescape
SN preference is low-z-driven — a low-z/covariance effect, not a high-z cosmological signal.

**T1 rider — free the void history → the split is timescape's rigidity.** A free spline E(z) fit
jointly to SN + DESI BAO + Planck CMB reaches χ²=1391.8 — it does **not** significantly beat ΛCDM
(1402.2; Δ=10.4 for 5 extra params, p=0.065) and both sit far below the timescape tracker (1469.3,
Δ=+67). (LCDM-matched start reproduces the committed ΛCDM χ²: 1404 vs 1402 — sanity gate passed.)
So SN+BAO+CMB *are* reconcilable by a smooth history: the split is timescape's **one-parameter
rigidity**, one tracker history unable to serve both the SN and the BAO+CMB shapes — not a model-
independent data conflict. (The disjoint-shell f_v0 fit railed 4/6 — narrow shells lack the baseline
to constrain the shape, and ΛCDM Om rails identically — so it is reported with that caveat and is
not load-bearing.)

## Synthesis against T1 / T2 / T3

| Sense | Verdict | Evidence |
|---|---|---|
| **T2 directional averaging** | **Rejected as the cause** | Probe 2 null (−0.9σ vs rotation null); Probe 3 f̂_v0 unmoved (Δχ²=0.6, p=0.55); Probe 5 effect ≤0.05 mag |
| **T3 external consistency** | **SN corner disfavoured** | Probe 4: 0.85 exceeds the observed void bracket [0.50, 0.62]; 0.64 plausible |
| **T1 temporal rigidity** | **The culprit** | T1 rider: free smooth history ≈ ΛCDM ≪ timescape tracker (Δ=+67) |

The timescape SN-vs-BAO+CMB tension is **not** because it ignores measured line-of-sight structure
(directional averaging adds nothing at current precision) and **not** an irreducible SN-vs-BAO shape
disagreement (a free smooth history reconciles them). It is the rigidity of the single-parameter
tracker history, forced to serve a low-z SN shape (pulling f_v0 → 0.85, a value with no physical
void-fraction support) and a BAO+CMB shape (pulling → 0.64) at once. This is a robustness result
that strengthens, and does not overturn, the paper's "recasting, not resolution" thesis.

## Honest caveats

- Map depth limits the LOS covariates to z ≲ 0.067 (573/1580 SNe); the LOS test therefore has no
  leverage on the high-z SNe that pin f_v0 — but that is exactly Probe 5's point (the effect is a
  low-z ceiling).
- Placement uses zCMB (not zHD) to avoid the 2M++ peculiar-velocity double-count; the reconstruction
  is FLRW-based, so the covariates inherit a fiducial-cosmology circularity that is second-order at
  these redshifts.
- Probe 3 is a phenomenological coupling ansatz, not derived GR; the Green–Wald critique applies to
  any such derivation. The null result is robust to this because the covariate carries no
  information regardless of the coupling's functional form (Probe 2, model-independent).
- Probe 4 is definition-dependent literature synthesis; the load-bearing claim (0.85 fails the most
  permissive void definition) is robust to the definitional ambiguity.

## Adversarial verification

A 5-agent independent re-derivation pass (each skeptic re-implemented the probe's statistics from
scratch and tried to refute the headline). Verdicts:

- **Probe 1 — CONFIRMED.** Hand-rolled trilinear interpolation reproduced F_i and mean_delta for
  all 573 rows to zero difference; G1 gate reproduced bit-for-bit. Noted (correctly) that the G1
  gate is a *radial* check and cannot catch an axis-swap — orientation correctness rests on Probe
  0's named-structure checks (Virgo/Coma/Local Void), which pass.
- **Probe 2 — CONFIRMED.** GLS slopes reproduced exactly; an independent rotation null with a
  different seed gave −0.86σ / p=0.47 (vs committed −0.91σ / p=0.40), so the wide null is
  seed-robust, not cherry-picked. Flagged that the naïve slope is wrong-sign vs pre-registration
  (folded into the text above) and that "artifact" is better phrased as "sub-threshold / not robust
  to a coherence-preserving null" (done).
- **Probe 3 — CONCERN, fixed.** The verifier found the coarse λ grid (step 0.025) straddled the
  true shallow minimum at λ≈−0.012, spuriously reporting Δχ²=0 and *structurally flooring* the
  rotation-null p to 1.0. Refit on a fine grid: λ̂=−0.0125, Δχ²=0.63, rotation-null p=0.55 — still a
  clean null, f̂_v0 still unmoved (fix committed; numbers above are the corrected ones).
- **Probe 5 — CONFIRMED.** Every number reproduced from scratch. Flagged that the ceiling used the
  naïve σ_λ; folding in Probe 3's rotation-null width raises the 3σ ceiling to |λ|<0.058 and RMS δμ
  to ≤0.046 mag (done) — still ≪ the error budget, conclusion unchanged.
- **T1 — CONFIRMED.** 46 Nelder-Mead restarts plus differential evolution all converged to the same
  free-shape minimum (1391.85) — the optimizer is not stuck near its LCDM start; p=0.065 verified.
  Noted (correctly) that LCDM is not exactly nested in the spline family, so the Wilks p is an
  approximation (conservative) and the p=0.065 verdict is borderline but robust.

No headline conclusion changed under adversarial re-derivation; one grid artifact (Probe 3) and two
error-calibration refinements (Probe 2 wording, Probe 5 ceiling) were corrected.
