# Can eliminating dark energy resolve the Hubble tension?

**A uniform test of void and backreaction cosmologies against supernovae, baryon acoustic oscillations, and the CMB acoustic scale.**

Author: **Viacheslav Zhygulin**

Paper (LaTeX source + compiled PDF) and full reproduction code for an independent test of whether *dark-energy-free* inhomogeneous cosmologies — David Wiltshire's **timescape** model and the wider void / backreaction family — can resolve the [Hubble tension](https://en.wikipedia.org/wiki/Hubble_tension).

> **Status:** working draft / pre-print, not yet submitted to arXiv. Shared for open review and reproduction. The analysis and drafting were AI-assisted (disclosed in the paper's Acknowledgements).

📄 **Read the paper:** [`timescape-hubble-tension.pdf`](timescape-hubble-tension.pdf)

## Findings in one paragraph

Within timescape the early- and late-time *dressed* Hubble constants reconcile internally at **H₀ ≈ 61 km/s/Mpc** — dissolving the *premise* of the tension (that there is a single FLRW H₀ to disagree about), but at a value *below* Planck's 67.4, i.e. a recast of the scale rather than a climb to the local ~73. Run through one uniform supernova + BAO + CMB harness, **no dark-energy-free void or backreaction model fits all three datasets with a single parameter set**: every model shows the same split — supernovae prefer a deeper void (void fraction f_v ≈ 0.85) than the early-Universe geometry allows (f_v ≈ 0.64) — and ΛCDM is statistically preferred. The preference against timescape is **≈1–2σ, not a decisive falsification**; its proponents report Bayesian *preference* for timescape using a cosmology-independent supernova covariance that is not publicly released, which this analysis can neither reproduce nor refute. Recalibrated TRGB distances (CCHP) put H₀ ≈ 68–70, only ~1–1.5σ from Planck, so the headline 5σ tension is largely specific to the SH0ES Cepheid ladder.

### Model comparison

Joint SN + BAO + CMB fit, ΔBIC relative to ΛCDM (larger = more disfavoured):

| Model | Dark-energy-free | ΔBIC | Verdict |
|---|---|---|---|
| ΛCDM (reference) | no | 0 | — |
| w₀wₐCDM (evolving DE) | no | +11 | mild DESI-direction hint, not significant |
| Timescape | yes | ~+67 | does not resolve |
| LTB giant void (GBH) | yes | ~+57 | does not resolve (+ kSZ-excluded) |
| Räsänen peak / backreaction | yes | ~+65 | predicts no acceleration |
| Buchert "morphon" | yes | ~+1600 | fails BAO+CMB |
| Szekeres | yes | intractable | does not resolve (literature + kSZ) |
| KBC local void | no (keeps DE) | ~0 | needs an unobserved deep void |

## Repository contents

| Path | What |
|---|---|
| `timescape-hubble-tension.tex` / `.pdf` | the paper (RevTeX, 7 pp.) |
| `REPORT.md` | extended research notes |
| `src/harness.py` | shared SN+BAO+CMB fitting harness (run its self-test first) |
| `src/fit_timescape.py`, `src/timescape_baocmb.py` | timescape distance model + BAO/CMB calibration |
| `src/joint_w0wa.py` | joint fit: ΛCDM vs. w₀wₐCDM vs. timescape |
| `src/fit_tripp.py`, `src/fit_robust.py` | cosmology-independent SN reduction + covariance-sensitivity |
| `src/model_*.py` | the wider void / backreaction family (LTB, Räsänen, Buchert, Szekeres, KBC) |
| `src/fig_*.py`, `src/final_analysis.py` | figure generation |
| `src/data/` | Pantheon+ supernova data + covariance (public release) |
| `results_*.json`, `fig_*.{pdf,png}` | numerical outputs and figures |

## Reproducing the results

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src
python harness.py          # self-test: ΛCDM joint fit (expect Om~0.305, H0~68.6)
python joint_w0wa.py       # ΛCDM vs w0waCDM vs timescape -> ../results_joint.json
python timescape_baocmb.py # timescape inverse-distance-ladder vs DESI BAO + CMB
python fit_robust.py       # SN covariance-sensitivity decomposition
```

Scripts read from `data/` and write results/figures to the repository root, so run them from inside `src/`.

Build the paper:

```bash
pdflatex timescape-hubble-tension.tex   # run twice for cross-references
```

## Data

The supernova inputs are the public **Pantheon+** release (Scolnic et al. 2022; Brout et al. 2022), redistributed here under `src/data/` for convenience. BAO are from **DESI DR1** (2024) and the CMB enters through the **Planck 2018** acoustic scale. Full references are in the paper's bibliography.

## Citation

> V. Zhygulin, *Can eliminating dark energy resolve the Hubble tension? A uniform test of void and backreaction cosmologies against supernovae, baryon acoustic oscillations, and the CMB acoustic scale* (2026). Working draft.

## License

No license is attached yet — all rights reserved by default. The repository is shared for review and reproduction; open an issue if you would like to reuse the code or figures.
