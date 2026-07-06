# Plan: improving "Can eliminating dark energy resolve the Hubble tension?"

*Consolidated 3-phase improvement plan, 2026-07-05. Built from a 12-agent audit/probe run
(5 code+paper audits, pipeline reproduction, literature sweep, 5 statistical-upgrade probes —
all complete; the adversarial verification pass was stopped early: only 1 of ~17 verdicts done).
Probe scripts live in `src/probes/`, machine-readable outputs in `probes_out/`, on this branch
(`significance-audit`). **All probe numbers below are single-sourced and unverified unless marked
CONFIRMED — the verification pass (§U1) must run before any number enters the paper.**

---

## A. What the audit established (inputs to the plan)

**A0. The pipeline reproduces.** Fresh venv (`requirements.txt`), all committed scripts rerun:
zero mismatches beyond rounding against every committed `results_*.json`. Data cuts, covariance
load (1701² symmetric, subset PD), DESI DR1 rows (exact match to arXiv:2404.03002 incl. all five
D_M–D_H correlations), dressed-distance implementation (H(z) verified to be the dressed
γ̄H̄ − dγ̄/dt on the tracker), offset/α profiling algebra, and BIC arithmetic all check out.
The paper's Sec. III tension arithmetic and Table I/II numbers reproduce exactly.

**A1. Status change in the literature (biggest single input).** The Seifert et al.
cosmology-independent covariance **is now public**: full pipeline incl. covariance construction at
`github.com/antosft/SNe-PantheonPlus-Analysis` (`BuildPP.py`, `how_to_covariance.ipynb`; repo
updated 2026-05-11). The paper's central caveat — "not publicly released … we can neither
reproduce nor refute" (abstract, §IV.D, §VII) — is **false as of 2026-07-05** and the paper's
top-listed decisive-test requirement (i) is now executable. Also: Seifert et al. quote
f_v0 = 0.737 ± 0.029 (z ≥ 0.055); Lane et al. full-sample MLE f_v0 = 0.6751; DESI DR2 BAO is
published with an official Gaussian likelihood (arXiv:2503.14738; Cobaya `desi_bao_dr2`).
New 2025–26 refs to fold in: Camarena (arXiv:2507.17969), Stiskalek et al. (2506.10518,
2509.14997, 2509.09665), Banik & Kalaitzidis (2501.17934), Banik et al. review (2602.03928),
Galoppo et al. (2511.17160, 2512.16591 — the Wiltshire-group CF4 answer to PV criticisms).

**A2. Defects found (bugs ▲ / concerns ●), each with location:**

| # | Sev | Finding | Where | Effect on paper |
|---|---|---|---|---|
| F1 | ▲ | Sec. V "χ²/dof=168 … at 0.737 it is 25" reproduces only under the superseded 0.40 CMB error; the corrected 0.05 pipeline (used for every other number in the same paragraph) gives **238 and ~39**; "25" also collides with the differently-defined committed 25.75 | tex:307; `timescape_baocmb.py` vs `verify_and_extend.py:21` | understates claim C; internally inconsistent paragraph |
| F2 | ▲ | "~1–2σ after CMB f_v0 systematic" has no committed arithmetic; reconstruction shows it holds only for the external 0.737 anchor (1.24σ); against the paper's own SN fit it is 2.33σ; and grafting Nazer's CMB-power-spectrum systematic onto the geometric BAO-shape fit contradicts the paper's own r_d-independence argument | tex:71–73, 311–312 | the central adverse finding is understated (see P1/P2 probes: ≥3.1σ, headline 4.4σ) |
| F3 | ▲ | LTB Table III "≈+57", "central H0 ~66" irreproducible: committed `model_ltbvoid.py` prints **+717.4**, H0 55–59 | tex:388, 426 | Table III row unsupported (verdict unchanged in sign, much stronger) |
| F4 | ▲ | Rasanen joint dBIC omits the +ln N penalty for its 2nd parameter (comment says "k equal"; LTB/w0wa apply it) → +65 should be **+72.5**; Table III internal ranking currently inconsistent | `model_rasanen.py:261` | corrected, Rasanen ranks below timescape |
| F5 | ▲ | `results.json` standard_full f_v0 upper error clipped at the 0.900 grid edge (true Δχ²=1 crossing 0.9025 → +0.0497 not +0.0471) | `final_analysis.py:19`, `fit_timescape.py:128` | minor; fix grid to ≥0.96 |
| F6 | ● | Raw-Tripp full-cov cell (+3.9): m_b_corr covariance applied to a Tripp magnitude vector (metric mismatch), with (α,β) frozen from the stat-only *timescape* fit; profiling (α,β) under full cov gives **+2.5**, canonical BBC (0.148, 3.09) gives +9.05 — sign never flips | `fit_robust.py:17,55` | "LCDM favoured under EVERY magnitude standardisation" needs the +2.5..+9.0 range and a covariance-mismatch caveat |
| F7 | ● | Abstract says +9.0 = "bias correction removed" (correct); body says "added back" (reads as the opposite); +9.0 in no results artifact; Table I omits the row | tex:58 vs 226 | wording fix + persist artifact |
| F8 | ● | Fig. `fig_fvtension` + `results_fvtension.json` built with superseded 0.40 CMB error (f_v0=0.639) while text uses 0.05 (0.636) | `fig_fvtension.py:14` | regenerate figure (direction: conservative today) |
| F9 | ● | "f_v0 = 0.636 ± 0.007": ±0.007 has no committed provenance (probe recomputation: ±0.0056; conservative) | tex:299 | persist recomputation |
| F10 | ● | CMB point mixes Planck columns: r*=144.39 (no-lensing) with r_d=147.09 (+lensing); consistent pairing shifts D_M(z*)/r_d 94.29→94.32 (~0.5σ of adopted error) | `verify_and_extend.py:17`, `timescape_baocmb.py:69` | recompute from one column |
| F11 | ● | Pipeline-validation checks (ii) F′(t) and (iii) low-z Taylor exist only as a hardcoded prose string; no committing script (audit re-verified (ii) analytically — it's true, just not demonstrated) | tex:210–217, `final_analysis.py:44` | write `src/probes/validate_ii_iii.py`, commit |
| F12 | ● | KBC "free joint fit drives the void to zero" never computed (only fixed literature void, dBIC +165, and the void=0 endpoint) | tex:402, `model_kbcvoid.py` | compute the free fit or rescope the sentence |
| F13 | ● | Buchert "~+1600" reruns to +1558.5; no artifact | tex:396 | regenerate + persist |
| F14 | ● | Table II q0 values: timescape −0.03 should be −0.02 (computed −0.0227); LCDM −0.55 taken from the BAO+CMB-only fit, not the joint row (joint gives −0.54) | tex:354–356 | rederive from `results_joint.json` |
| F15 | ● | Standalone `fit_timescape.py` grid caps at f_v0=0.799 → railed (+6.8 vs paper's +5.1); committed `scan_results.npz` is that railed artifact — **CONFIRMED by verification** | `fit_timescape.py:138` | widen grid; regenerate npz |

**A3. Probe results (unverified; ready to upgrade the paper once checked):**

- **P1 `fvsplit`** — profile-likelihood parameter-shift statistic (Δχ²_join, 1 dof) over a 5 SN-reductions × 4 BAO-treatments matrix. Headline: **SN(standard, full cov) vs DESI-DR1 BAO-ONLY = 4.4σ** — a pairing free of the CMB point, the r_d calibration, and every CMB systematic. Floor across all SN reductions: 3.1σ (Tripp z>0.075). The "~1–2σ" is reproducible **only** as (external 0.737 anchor) + (Nazer 13% grafted onto the geometric fit). Full matrix + corrected unclipped errors in `probes_out/fvsplit.json`; suggested replacement wording in its `conclusion` field.
- **P2 `mocks`** — parametric bootstrap, 4000 mocks/truth, full stat+sys covariance. SN-only: observed ΔBIC=+5.14 sits at the 25th percentile of LCDM-truth (median +9.2) and has p=0.008 (2.4σ one-sided) under timescape-truth → Sec. IV can state "SNe alone reject timescape at ~2.4σ". f_v split: observed 0.214 exceeded by **0 of 4000** single-f_v0 mocks (largest 0.122) → distribution-free ≥3.2σ, Gaussian/GPD tail ~7.5σ (extrapolated — quote the floor, footnote the tail).
- **P3 `evidence`** — exact 1-D quadrature Bayes factors (removes the "BIC is crude" objection). Full-cov standard cell: ln B(ΛCDM/TS) = +1.31..+1.80 across a 2×2 prior grid (Kass–Raftery "positive"; smaller than ΔBIC/2 because ΛCDM pays a ~1-nat larger Occam factor — worth a footnote). Stat-only Tripp corner: ln B = **5.38..5.91 for timescape** ("very strong") — i.e. the paper's own machinery reproduces the *magnitude and direction* of Seifert's ln B > 5 purely from the covariance treatment. This is a stronger structural point than the current ΔBIC=−9.4 sentence.
- **P4 `dr2`** — DESI DR2 (official Cobaya likelihood; cross-validated: BAO-only ΛCDM refit Ω_m=0.2975±0.0086 = DESI's published value). f_v0 unchanged (0.6355 BAO+CMB; 0.677 BAO-only) but σ(f_v0) shrinks 1.7× (+0.0031/−0.0035); timescape χ²/dof worsens 3.30→**6.67** while ΛCDM improves 1.18→**0.89**; BAO+CMB-alone ΔBIC(TS−ΛCDM) +23.3→**+69.3**; split vs own SN fit: **6.6σ** (BAO+CMB), 4.8σ (BAO-only); vs Seifert anchor: 3.5σ (BAO+CMB), 1.95σ (BAO-only floor). DR1 framing is now conservative on every axis.
- **P5 `freshH0`** — the paper's decisive-test (ii), executed: SH0ES-calibrator-anchored fit inside the paper's own machinery (gate passed: ΛCDM H0=73.53±1.0, Ω_m=0.333 vs Brout et al. 73.5±1.1). Fresh late-time **timescape dressed H0 = 73.00 ± 1.00** (f_v0=0.853, M_B=−19.24) — not 61.7. Claim A's ≤0.5σ agreement becomes **9.4σ (stat) / 2.4σ (stat⊕sys)** vs the CMB-calibrated 61.0 — worse than the ΛCDM analogue (5.4σ) on identical data. δ_req = 0.197±0.023 still lands inside the predicted 17–22% window, so claim A survives *only* through the expansion-variance argument, never as an H0 agreement. This decisively sharpens "recasting, not resolution".

---

## Phase 1 — Correct and sharpen the statistics on current data

1. **Fix the five ▲ bugs and ten ● concerns** (table A2), in code first, then tex. Every number
   quoted in the paper must have a committed generating script + results artifact (F1, F3, F7,
   F9, F11, F12, F13 are provenance gaps; F4, F5, F15 are code fixes; F2 is a claims fix).
2. **Replace the "~1–2σ" significance statement** (tex:71–73, 311–312) with the systematics
   ladder from P1, using the BAO-only pairing as the headline (CMB/r_d-free, 4.4σ) and the
   3.1σ floor; keep the 0.737-anchor + Nazer-systematic case only as an explicitly-scoped
   worst-case sentence. Suggested wording sits in `probes_out/fvsplit.json → conclusion`.
3. **Add the frequentist calibration** (P2): the ~2.4σ SN-only statement for Sec. IV and the
   0/4000 (≥3.2σ distribution-free) split exclusion for Sec. V.
4. **Add exact evidences** (P3): one paragraph + small table; footnote the Occam-factor
   asymmetry; make the structural point that the stat-only Tripp corner reproduces Seifert's
   ln B > 5 in-house.
5. **Regenerate figures** with the corrected CMB error (F8) and unrailed grids (F5/F15).
6. Rebuild Table III with consistent BIC accounting (F3, F4, F12, F13) and persist per-model
   results files.

## Phase 2 — Recalculate on fresh measurement data

1. **Swap DESI DR1 → DR2** (P4) throughout Sec. V and Table II/III; keep DR1 in one
   sensitivity sentence. All DR2 rows + covariance are recorded in `probes_out/dr2.json`
   (source: Cobaya `desi_bao_dr2` @ b7b8a36).
2. **Reproduce the Seifert covariance** from the now-public `antosft/SNe-PantheonPlus-Analysis`
   (A1) and rerun Sec. IV under it: this converts the paper's central "cannot reproduce" caveat
   into an actual test, settles the sign of the supernova preference, and rewrites §IV.D,
   the abstract, and decisive-test (i). *This is the highest-value single task in the plan.*
3. **Add the fresh calibrator-anchored H0 section** (P5) as the paper's own execution of
   decisive-test (ii); reframe claim A accordingly (agreement-at-61 is calibration-relative;
   the fresh anchored fit gives 73.0±1.0 and the recasting interpretation carries the full load).
4. **Refresh the literature context** with the 2025–26 refs (A1), especially Galoppo et al.
   2512.16591 (proponents' PV answer) and Camarena 2507.17969 (methodological counterpoint).
5. Optional cross-checks if scope allows: DES-SN5YR / Union3 as alternative SN samples;
   consistent-column Planck CMB point (F10).

## Phase 3 — Beyond the single-parameter average (line-of-sight voids)

Execute `PLAN_los_voids.md` (committed on this branch): Probe 0 (data feasibility gate,
2M++/Carrick field), Probe 1 (per-SN LOS void statistics), Probe 2 (GLS regression of residuals
on LOS void fraction; pre-registered sign; zCMB not zHD), Probe 3 (one-parameter per-sightline
timescape extension — does f̂_v0 move toward 0.64?), Probe 4 (external f_v0 bracket vs the
0.85/0.64 demands), Probe 5 (analytic z>0.1 ceiling), and the T1 rider (nonparametric spline
D(z) fit separating "timescape is rigid" from "SN and BAO shapes disagree model-independently").
Outcome feeds a new discussion subsection (or a companion paper) answering whether the failure
is the *averaging* or the *void hypothesis itself*.

## Phase 4 — void-population-forced backreaction (designed 2026-07-06, not executed)

Execute `PLAN_void_history.md`: replace the tracker closure with the *observed* void
history f_v(z) (Model V) and depth/size distribution (V2) through the same Buchert
averaging + lapse dressing; Probe R (required-vs-available void history) gates the rest.
Answers "right mechanism, wrong parameter set?" after T1 located the failure in tracker
rigidity. Publication routing (same paper vs companion) is decided after Probe R.

---

## U. Unfinished work (delegation list)

- **U1. Adversarial verification pass (do FIRST — blocks everything above).** Only
  `standalone-script-railed` (F15) is CONFIRMED. To verify: the 14 remaining ▲/● findings in
  table A2 (protocol: read the cited code/tex, redo the arithmetic independently, verdict
  CONFIRMED/REFUTED with corrected statement) and the 5 probes P1–P5 (protocol: read the probe
  script line-by-line for statistical errors — wrong tail, dof, covariance misuse, grid
  coarseness, prior sensitivity, mock-generation mismatch — then independently spot-check one
  headline number each). Scripts: `src/probes/{fvsplit,mocks,evidence,dr2,freshH0}.py`; outputs
  in `probes_out/*.json`; a working venv recipe is `python3 -m venv .venv && pip install -r
  requirements.txt` (numpy 2.0.2 / scipy 1.13.1).
- **U2. Seifert-covariance reproduction** (Phase 2.2) — clone `antosft/SNe-PantheonPlus-Analysis`,
  build the cosmology-independent covariance per `how_to_covariance.ipynb`, run the paper's
  Sec. IV grid under it, and report ΔBIC/ln B + f_v0. Expect nontrivial effort (their pipeline
  rebuilds the covariance from SALT2 fits; PyMultinest optional — the paper only needs the
  covariance + the existing grid machinery).
- **U3. Phase 3 execution** (`PLAN_los_voids.md`) — external data acquisition is the risk;
  gates and fallbacks are specified there.
- **U4. Paper edits** — apply Phases 1–2 to the tex only after U1 verdicts land; keep the
  one-number-one-artifact rule.
- **U5. DR2 regeneration of Table II/III** including the family models (currently DR1-only).

Suggested order: U1 → U2 → Phase 1 → Phase 2 (incl. U5) → U4 → U3.

---

## Assets on this branch

- `src/probes/fvsplit.py`, `mocks.py`, `evidence.py`, `dr2.py`, `freshH0.py` + outputs in
  `probes_out/*.json` (machine-readable; each has a `conclusion` field with suggested wording).
- `probes_out/orig_snapshot/` (pre-run results), `probes_out/regen/` + `run_*.log`
  (reproduction evidence), `compare_results.py` (the zero-mismatch check).
- `PLAN_los_voids.md` (Phase 3 detail).
- Workflow journal (agent-by-agent returns): session transcript dir
  `subagents/workflows/wf_567e39c9-0bd/journal.jsonl`.
