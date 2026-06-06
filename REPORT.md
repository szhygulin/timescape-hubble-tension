# Does the Timescape model resolve the Hubble tension?

*Research + computation report. Generated autonomously 2026-06-05. Topic: testing the hypothesis that the Hubble tension ("напруження Хабла") dissolves under a dark-energy-free inhomogeneous-expansion cosmology.*

---

## TL;DR — the two numbers under the new model

The model you described is **Timescape cosmology** (David Wiltshire, University of Canterbury) — no dark energy, no Λ; apparent acceleration is a general-relativistic artifact of clocks running faster in empty voids than in dense galaxy "walls." The Dec-2024 paper that fits the supernova data well is [Seifert, Lane, Galoppo, Ridden-Harper & Wiltshire 2024/2025, *MNRAS Letters* 537, L55 (arXiv:2412.15143)](https://arxiv.org/abs/2412.15143).

**The two Hubble numbers, recomputed within Timescape (the "dressed" Hubble constant a galaxy observer infers):**

| Probe | ΛCDM (the tension) | Timescape (dressed H₀) |
|---|---|---|
| **Indirect** (early / CMB) | 67.36 ± 0.54 | **61.0 km/s/Mpc** (±1.3% stat, ±8% sys) |
| **Direct** (late / Type Ia SNe) | 73.04 ± 1.04 | **61.7 km/s/Mpc** (⁺¹·²₋₁.₁) |
| **Gap** | 5.68 km/s/Mpc ≈ **~5σ** | 0.7 km/s/Mpc ≈ **0.1–0.5σ** |

**Headline:** when both the early- and late-universe data are interpreted *inside* Timescape (dropping the homogeneous-FLRW assumption), they return the **same** Hubble constant, ≈ 61 km/s/Mpc, agreeing to well within 1σ. The ΛCDM 5σ tension disappears. **In that sense your hypothesis holds: Timescape has no Hubble tension.**

**But — important correction to the framing (this is the part most popular accounts get wrong):** Timescape does **not** make 67 and 73 meet near 70. It makes them both collapse to ≈ **61, which is *below* even Planck's 67.4.** The locally-measured 73 is not recomputed to 61 — it is *reinterpreted* as a real, expected **scale-dependent measurement artifact**: we live in a dense wall, nearby voids expand faster, so a local observer fitting a single Hubble law infers a value biased high (up to ~+17%) until the survey reaches the ~100 h⁻¹ Mpc statistical-homogeneity scale, where it settles to the true dressed 61.7. So the verdict is **RECAST, not number-matched-resolved.**

---

## The mechanism (why no dark energy is needed)

- The late universe is ~60% voids by volume. Timescape models it as two phases: spatially-flat **walls** (finite-infinity regions around galaxies, where we sit) and negatively-curved **voids** that dominate the volume and expand faster (in the tracker limit, as empty Milne regions).
- Gravitational energy and curvature gradients between walls and voids cannot be localized (equivalence principle) ⇒ a genuine **quasilocal clock-rate difference**: void clocks run ~17–38% faster than wall clocks at the present epoch (present lapse γ̄₀ ≈ 1.31–1.38).
- A wall observer who naively fits a homogeneous FLRW model gets **"dressed" parameters** that differ from the true volume-average **"bare" parameters**. The *bare* deceleration parameter stays **positive** (no real acceleration; ordinary matter only), yet the *dressed* one goes **negative** — apparent acceleration is a calibration artifact, not a repulsive force. No dark energy.

Foundational papers: [Wiltshire 2007, *PRL* 99, 251101 (arXiv:0709.0732)](https://arxiv.org/abs/0709.0732); [Wiltshire 2009, *PRD* 80, 123512 (arXiv:0909.0749)](https://arxiv.org/abs/0909.0749).

---

## The computation

### Verified governing equation (adversarially confirmed verbatim)

The tracker-solution relation between the dressed (locally-inferred) and bare (volume-average) Hubble constants — Wiltshire 2009 **eq. (B9)** = Wiltshire 2007 PRL **eq. (25)**:

```
H₀(dressed) = [ (4·fv₀² + fv₀ + 4) / (2·(2 + fv₀)) ] · H̄₀(bare)
```

(`fv₀` = present void volume fraction. The middle term is `+fv₀`, coefficient 1 — a "4·fv₀" middle term seen in some secondary write-ups is a transcription error.)

Present-day lapse (eq. B7): `γ̄₀ = (2 + fv₀)/2`. Note dressed H₀ is **not** simply γ̄₀·H̄₀ — the general relation (eq. 28) `H = γ̄·H̄ − dγ̄/dt` carries a lapse-rate correction, so the algebraic ratio (1.23–1.28) is smaller than γ̄₀ (1.31–1.38).

### Step-by-step

**Step 1 — ΛCDM tension significance.** 73.04 − 67.36 = 5.68 km/s/Mpc. Combined error √(1.04² + 0.54²) = 1.172. Naive significance 5.68/1.172 = **4.85σ**; SH0ES fold in extra systematics to quote **5σ**.

**Step 2 — Bare→dressed, Planck-constrained fit (fv₀ = 0.695, H̄₀ = 50.1; [Duley, Nazer & Wiltshire 2013, arXiv:1306.3208](https://arxiv.org/abs/1306.3208) Table 1).**
Numerator = 4(0.695²) + 0.695 + 4 = 1.9321 + 0.695 + 4 = 6.6271.
Denominator = 2(2.695) = 5.390. Ratio = **1.2295**.
Dressed H₀ = 1.2295 × 50.1 = **61.6 km/s/Mpc** ✓ (matches published 61.7).

**Step 3 — Sanity check, canonical fv₀ = 0.762, bare 48.2.** Ratio = 7.0846/5.524 = 1.2825 → 1.2825 × 48.2 = **61.8** ✓. Both independent fits land the dressed value at ≈ 61.7.

**Step 4 — Lapse.** γ̄₀ = (2+0.695)/2 = 1.348 (matches Duley 2013); = 1.381 for fv₀=0.762; = 1.314 for fv₀=0.627.

**Step 5 — Early-universe (CMB) dressed H₀.** Full Planck-multipole MCMC ([Nazer & Wiltshire 2015, *PRD* 91, 063519, arXiv:1410.3470](https://arxiv.org/abs/1410.3470)): **H₀ = 61.0**, ±0.79 stat (±1.3%), ±4.88 sys (±8%); fv₀ = 0.627.

**Step 6 — Late-universe (SNe) dressed H₀.** Canonical SNe-era best fit (Leith-Ng-Wiltshire 2008 / Wiltshire 2009): **61.7** (⁺¹·²₋₁.₁); Planck-constrained refinement (Duley 2013): 61.7 ± 3.0.

**Step 7 — Early-vs-late consistency.** Residual 61.7 − 61.0 = 0.7 km/s/Mpc. Stat-only combined error √(0.79² + 1.15²) = 1.40 → **0.5σ**; with full systematics → ~0.1σ. Agreement well within 1σ.

**Step 8 — Verdict.** No internal early/late tension in Timescape (the two dressed values coincide at ~61). The ΛCDM 5σ gap is reinterpreted as scale-dependent local variance + rest-frame choice ([Wiltshire, Smale, Mattsson & Watkins 2013, arXiv:1201.5371](https://arxiv.org/abs/1201.5371) argue the Local-Group frame is preferred), not literally collapsed onto one number.

---

## How strong is the evidence? (honest scorecard)

**For:**
- [Seifert et al. 2024/2025 (Pantheon+, arXiv:2412.15143)](https://arxiv.org/abs/2412.15143): very strong Bayesian evidence for Timescape over ΛCDM on the **full** sample, **ln B > 5**.
- Companion [Lane et al. 2023/2025, *MNRAS* 536, 1752 (arXiv:2311.01438)](https://arxiv.org/abs/2311.01438): first Pantheon+ evidence Timescape may out-fit ΛCDM.
- The early/late dressed-H₀ internal consistency at ~61 (this report).

**Against / unsettled (Timescape is a serious but distinctly *minority* program):**
- That **ln B > 5 falls to only ln B > 1 (moderate)** once you cut to z > 0.075 (beyond the statistical-homogeneity scale); the strong signal is driven by low-z supernovae. The authors concede that above the homogeneity scale "ΛCDM provides an excellent description."
- The **newest SNe analyses fix H₀ as a nuisance parameter** (degenerate with SN absolute magnitude) — so Pantheon+ does **not** measure a fresh dressed H₀. The 61.7 "late" number is the older SNe-era fit.
- The **CMB fit is not first-principles**: Nazer & Wiltshire 2015 adapts an ΛCDM template via a scaling transformation, fits only the acoustic scale (50 ≤ ℓ ≤ 2500), assumes standard pre-recombination physics, and excludes the low-ℓ late-ISW region Timescape modifies. The 8% systematic dominates its H₀ error.
- **BAO + structure growth** (fσ₈, weak lensing) are far less developed than in ΛCDM; calibration flagged as an urgent open task (Euclid-era test).
- **Backreaction theory dispute:** Green & Wald argue backreaction is dynamically negligible (~10⁻⁵); Buchert et al. dispute the theorem's applicability. Unresolved.
- **Pantheon+ contestation:** the preference is sensitive to covariance construction and to omitting full FLRW peculiar-velocity corrections at low z (applying them is argued to shift preference back toward ΛCDM); fitter-dependent (SALT2 vs MLCS2k2).

---

## Assumptions / judgment calls made autonomously (no user present)

1. **Model identity inferred, not given.** You didn't name the model; I identified it as Timescape from your description (no dark energy + voids expand faster + recent paper fitting telescope data well). Confidence high — it matches the Dec-2024 Pantheon+ result exactly.
2. **"Two numbers" = direct vs indirect H₀.** I read your "two numbers" as the late/SNe and early/CMB Hubble constants — the standard framing of the tension you gave — and report the Timescape dressed H₀ for each.
3. **I did NOT run a new fit from raw data.** "Proper computations under the new model" = (a) the exact analytic bare→dressed tracker conversion with the verified equation, and (b) the tension-significance arithmetic. Re-fitting Timescape to raw SNe/CMB likelihoods needs the full pipeline + datasets (days of compute) and was out of scope. The dressed H₀ values are the **published best-fits**; the consistency/σ numbers are my own computation on top of them.
4. **Which fits I treated as authoritative.** Timescape has no single canonical H₀ from the newest data. I used the CMB fit (Nazer & Wiltshire 2015 → 61.0) and the SNe-era / Planck-constrained fits (→ 61.7) because those actually report a dressed H₀. The newest SNe paper fixes H₀, so the "late" number is the older fit — a limitation, not a fresh measurement.
5. **Void fraction fv₀ is fit-dependent (0.63–0.78); I did not pick one "true" value** — I showed the conversion for fv₀ = 0.695 and cross-checked 0.762, and flagged the spread. The dressed 61.7 is robust across fits only because each co-adjusts H̄₀ to land there.
6. **Framing verdict overrides your premise.** Your hypothesis said the tension "resolves." The primary sources support **recast**, not number-matching resolution (both go to ~61, *below* Planck; SH0ES 73 is reframed as a local-variance artifact). I reported the correction rather than just confirming the premise.
7. **Significance arithmetic is approximate.** σ values treat errors as independent Gaussians; the CMB 8% systematic isn't cleanly Gaussian and the fits share data lineage. "Consistent well within 1σ" is the defensible claim; don't over-read the exact 0.1–0.5σ.
8. **Source-access caveats.** Several arXiv PDFs failed text extraction; I used ar5iv HTML mirrors (cross-checked). The ~17% local-H₀-peak figure and a couple of secondary numbers came from search summaries, flagged. Leith-Ng-Wiltshire 2008 numbers were read partly via restatement in later papers.
9. **No 2025-2026 dedicated rebuttal** of Seifert et al. was found in this sweep; the published critiques are the methodological caveats above. (Knowledge horizon ~Jan 2026.)
10. **Saved here, not to the office KB.** This is personal research, so it lives at `~/research/timescape-hubble-tension/` — not the agenthill pkb-mcp office Drive.

---

## Bottom line

Under Timescape, the direct and indirect Hubble measurements both give a dressed **H₀ ≈ 61 km/s/Mpc** (61.0 from the CMB, 61.7 from supernovae), agreeing to within ~0.1–0.5σ versus the ~5σ ΛCDM gap. So the model **does dissolve the early-vs-late inconsistency** — your core intuition is supported. The caveat that matters: it does so by reinterpreting *both* canonical numbers as biased high relative to a true global value of ~61 (the local 73 being a scale-dependent artifact), not by reconciling them near 70 — and it remains a credible **minority** program whose CMB/BAO/growth pillars are not yet established to ΛCDM's standard.
