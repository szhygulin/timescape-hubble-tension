# Submission prep — Paper I (timescape Hubble tension)

**Prepare-only.** Minting the DOI and posting to arXiv are the author's to perform; nothing
here has been submitted. **Post this paper first** — Paper II's ref `[1]` needs this paper's
arXiv ID before Paper II goes up.

---

## 1. Zenodo deposit — `timescape-hubble-tension` (code + data record for Paper I)

- **Upload type:** Software
- **Title:** A uniform test of void and backreaction cosmologies against SN+BAO+CMB: code and data (Paper I)
- **Authors / creators:** Zhygulin, Viacheslav
- **License:** MIT
- **Version:** v1.0
- **Description (draft):**
  > Analysis code, committed numerical artifacts, and manuscript for *"Can eliminating dark
  > energy resolve the Hubble tension? A uniform test of void and backreaction cosmologies
  > against supernovae, baryon acoustic oscillations, and the CMB acoustic scale."* A single
  > harness scores the standard one-parameter timescape tracker (and comparison models)
  > against Pantheon+, DESI BAO, and the Planck acoustic scale, finding the tracker cannot
  > fit SNe and BAO+CMB together. Includes the reproduction of the Seifert et al.
  > cosmology-independent Pantheon+ covariance analysis. Every headline number is reproduced
  > by a committed probe script + JSON artifact and cross-checked by an adversarial pass.
  > Reproducibility: Python 3.12, numpy 2.0.2 / scipy 1.13.1; large external inputs (Seifert
  > P+1690 covariance, Zenodo 12729746) are fetched, not redistributed.
- **Keywords:** timescape cosmology; cosmological backreaction; Hubble tension; type Ia
  supernovae; baryon acoustic oscillations; CMB acoustic scale; inhomogeneous cosmology
- **Related identifiers:**
  - `isSupplementTo` → Paper I arXiv ID (once posted)
  - `isPreviousVersionOf` / `isContinuedBy` → `free-history-timescape` Zenodo DOI (Paper II)
  - `references` → Seifert P+1690 covariance, `10.5281/zenodo.12729746`

---

## 2. arXiv — Paper I

- **Primary category:** astro-ph.CO
- **Title:** Can eliminating dark energy resolve the Hubble tension? A uniform test of void
  and backreaction cosmologies against supernovae, baryon acoustic oscillations, and the CMB
  acoustic scale
- **Authors:** Viacheslav Zhygulin
- **Comments:** 10 pages. Paper I of a program; continued in arXiv:XXXX.XXXXX (Paper II).
  Code and artifacts: (Zenodo DOI once minted).
- **Abstract:** use the manuscript abstract **verbatim** — it is 1862 characters, within the
  arXiv field limit; no condensation needed.

---

## Post-submission stitch (do after Paper I is live)

1. Record Paper I's arXiv ID.
2. In `free-history-timescape`: replace ref `[1]` (`\bibitem{Paper1}` "(2026), companion
   paper…") with the arXiv-ID'd citation, rebuild the PDF (`tectonic
   free-history-timescape.tex`), commit, and only then post Paper II.
3. Add each paper's arXiv DOI as a `isSupplementTo` related identifier on its Zenodo record.
