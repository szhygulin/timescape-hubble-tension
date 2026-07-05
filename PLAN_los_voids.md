# Calculation plan: line-of-sight void data vs the single-parameter timescape average

*Plan only — execution delegated. Prepared 2026-07-05 in the `significance-audit` worktree.
Companion machinery (mocks, f_v profile-likelihood curves, venv) is produced by the
`timescape-significance-audit` workflow under `src/probes/` and `probes_out/`.*

## Question being tested

Timescape compresses all inhomogeneity into one global scalar `f_v0` (present void
volume fraction) whose time evolution is locked to a one-parameter tracker attractor,
and applies the same average distance–redshift curve to every sightline. But the local
large-scale structure is *measured* (at low z). Does injecting the measured
line-of-sight (LOS) void content per supernova (a) reveal signal the average model
misses, and (b) dissolve or deepen the SN-vs-BAO+CMB void-fraction split
(f_v = 0.85 vs 0.64) that currently disfavours timescape?

Three distinct senses of "the model is too simple", each testable:

- **T1 (temporal rigidity):** f_v(t) is locked to the tracker. Could *any* smooth
  void-history fit SN and BAO+CMB together?
- **T2 (directional averaging):** every SN gets the average curve; the actual LOS void
  fraction varies SN-to-SN, strongly so at z ≲ 0.1.
- **T3 (external consistency):** the SN fit *demands* f_v0 ≈ 0.85; void catalogs
  measure the actual void volume fraction — is 0.85 even allowed?

## Data inventory

In-repo (`src/data/PantheonSH0ES.dat`, 1701 rows, 1580 cosmology-sample SNe):
`RA`, `DEC`, `zHD`, `zCMB`, `zHEL`, `VPEC`, `VPECERR`, `m_b_corr`, `IS_CALIBRATOR`,
`biasCor_m_b`, full 1701² STAT+SYS covariance (`PantheonSH0ES_STATSYS.cov`).

**Critical caveat baked into every probe:** `zHD` already contains peculiar-velocity
corrections built from the 2M++ density reconstruction — the standard pipeline already
injects the known local structure, the FLRW way. Any LOS-structure covariate test MUST
use `zCMB` (or strip `VPEC`) on the left-hand side, else you regress data on itself.
(This is also exactly the axis on which Lane/Seifert dispute Pantheon+.)

External (acquire in Probe 0):
- Primary: Carrick et al. 2015 (arXiv:1504.04627) 2M++ galaxy density field δ_g —
  all-sky, usable to z ≈ 0.067. Public download (check current mirror; else BORG/2M++
  posterior means, or CosmicFlows-4 density).
- Void catalogs: VIDE/REVOLVER SDSS DR7 (z < 0.2), Mao et al. DR12 LOWZ/CMASS
  (0.2 < z < 0.7, SDSS footprint), DESI DR1 void catalog if public by execution time.
- Literature values of observed void volume fractions + Wiltshire's own operational
  wall/void split (finite-infinity definition), for the T3 mapping.

## Probe 0 — feasibility gate

Confirm one all-sky density field at z < 0.07 is obtainable. If not, prespecify the
footprint-restricted subsample and its power loss before fitting anything.
Count usable SNe (expect ~150–250 at z < 0.067 all-sky; more if SDSS-footprint
catalogs extend the range). **Gate: proceed only with a written sample definition.**

## Probe 1 — per-SN LOS void statistics (data engineering, no cosmology)

For each usable SN, sample the comoving path observer→SN through the field
(fiducial low-z distance conversion; circularity is second-order here) and compute:
- `F_i` = path-length fraction with δ below threshold (scan δ_th ∈ {−0.3, −0.5, −0.7});
- `mean_delta_i` = path-averaged density contrast;
- `delta_loc_i` = shell-averaged contrast at the SN's position (for expansion-variance
  coupling — the monopole around the SN, not just the beam, is what timescape-type
  models care about).

Validation gates: (G1) the sightline-averaged global void fraction must reproduce the
map's own global value; sanity sky-maps by survey.

## Probe 2 — model-independent GLS regression (cheap, decisive first test)

Regress Hubble residuals r_i = μ_obs(zCMB) − μ_model(z; best fit) on F_i (and
mean_delta_i), for BOTH μ_model ∈ {best ΛCDM, best timescape}, using the full
covariance (GLS slope λ̂ ± σ_λ), plus a permutation p-value (shuffle covariates among
SNe of similar z to kill redshift-trend confounds).

- Derive and pre-register the predicted sign before looking (more LOS void ⇒ faster
  local expansion ⇒ at fixed z the SN is nearer/brighter ⇒ sign of λ under the chosen
  residual convention — the executor must write this derivation down first).
- Confound controls: smooth f(z) nuisance term, host-mass step, survey ID.
- Negative control (G3): rotated-sky sham covariate must return null.
- Positive control (G4): repeating with zHD must visibly change the answer (if it
  doesn't, the covariate pipeline is broken — VPEC encodes this same structure).

Decision rule: |λ̂|/σ_λ ≥ 3 with the pre-registered sign ⇒ LOS structure carries
information beyond the average model → Probe 3 is mandatory. Null ⇒ the known voids
add nothing at current precision — the single-parameter average is NOT the bottleneck;
report as a robustness result (valuable either way).

## Probe 3 — phenomenological per-sightline timescape (one new parameter)

Two pinned extensions (do not invent further knobs):

- **A (expansion-variance coupling):** replace each SN's redshift by
  z_i → z_i + (1+z_i)·ΔH_i·D_i/c with ΔH_i/H = λ·(F_i − F̄). Fit (f_v0, λ, offset)
  with the full covariance.
- **B (mixture distance):** per-SN dressed distance from a perturbed void-history
  f_v^i(t) = tracker_{f_v0}(t)·[1 + λ (F_i − F̄) s(t)], s(t) ∝ f_v(t)/f_v0
  (growth-weighted, pinned).

Report for each: Δχ²(λ), λ̂ ± σ, BIC/evidence bookkeeping for one added parameter,
mock-calibrated p-value (reuse `probes_out` mock machinery), and — the headline —
whether the global f̂_v0 moves from 0.85 toward the BAO+CMB 0.64 (does measured
structure dissolve the split, or does it stay?). Gate G2: with λ = 0 the fit must
reproduce the repo's committed χ² exactly.

## Probe 4 — external void-fraction consistency (no fitting)

Map catalog/watershed void definitions onto Wiltshire's operational f_v (finite-infinity
walls vs voids; use Williams et al. 2024 numerical-relativity void statistics as the
bridge). Produce a defensible observed bracket f_v0^obs = [x, y] and place 0.85 (SN
demand) and 0.64 (BAO+CMB demand) against it. This is an independent, zero-fit data
point on which side of the split is unphysical.

## Probe 5 — analytic ceiling for z > 0.1 (arithmetic only)

Var(F_i) shrinks ~ L_void/D(z) (independent-cell counting, L_void ≈ 30–60 Mpc).
Convert to magnitudes via the Probe-3A coupling and show the maximal per-sightline
effect vs the Pantheon+ error budget (incl. σ_lens ≈ 0.055z already in the covariance)
as a function of z. Pre-registers the expectation that all discriminating power is at
z ≲ 0.1 — which independently adjudicates WHY the published timescape preference is
low-z-driven (real structure vs covariance artifact).

## T1 rider — free the void history (in-repo only, no external data)

Nonparametric check that separates "timescape is rigid" from "the data are internally
split": spline-parametrize the distance shape D(z) (≈8 nodes, E(z) > 0), fit jointly to
SN (offset profiled) + BAO (+CMB point, amplitude profiled), and compare
χ²_min(free shape) against χ²_min(ΛCDM) and χ²_min(timescape tracker).
- free ≈ ΛCDM ⇒ the split is timescape-specific rigidity (T1 confirmed as the culprit);
- free ≪ both ⇒ SN and BAO shapes disagree model-independently and NO smooth
  history (timescape generalizations included) reconciles them.
Also fit f_v0 in disjoint z-shells ({0.01–0.05, 0.05–0.1, 0.1–0.2, 0.2–0.4, 0.4–0.8,
0.8–2.3}, GLS with covariance sub-blocks) — the drift f̂_v(z) is the quantitative
statement of "one parameter doing two jobs" within SNe alone.

## Deliverables & discipline

- Every probe = one committed script under `src/probes/` + one JSON under
  `probes_out/` + the gate results. No edits to existing source files.
- Pre-registered signs/decision rules written before fitting (Probes 2, 3).
- Honest-caveats section in the final report: map depth (power limited to ~10–15% of
  the sample), void-definition mismatch, FLRW-fiducial circularity in catalogs,
  VPEC double-count avoidance, and that Probe 3 is phenomenology (a coupling ansatz),
  not derived GR — the Green–Wald critique applies to any such derivation.

## Expected outcomes (for calibration, not prejudice)

- Probe 2 at current sample sizes has σ_λ large; a null is the likely outcome and is
  informative (the average is sufficient at this precision).
- Probe 3 is unlikely to move f̂_v0 to 0.64 because the BAO+CMB side of the split is
  structure-map-independent; if λ absorbs the low-z pull, the SN preference for deep
  voids should *weaken*, further sharpening the case that the Seifert-type preference
  is a low-z/covariance effect.
- Probe 4 likely brackets f_v0^obs well below 0.85 under any defensible mapping —
  an independent strike against the SN-preferred corner.
