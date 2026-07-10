# Purpose: derive the non-parabolicity multiplier quoted in Sec. V ("the rise near
# the joint minimum is ~1.9x the Gaussian expectation") from the committed artifacts,
# probes_out/fvsplit.json and probes_out/dr2.json, with no refitting. For each
# SN-vs-BAO(+CMB) pairing the artifacts store the observed parameter-shift statistic
# Delta-chi2_join (the joint rise above the two separate minima) and the Gaussian
# parameter-difference significance sigma_gauss built from the two profile widths.
# Under exactly parabolic (Gaussian) profiles the joint rise would equal
# sigma_gauss^2, so
#     multiplier = Delta-chi2_join / sigma_gauss^2 = (sigma_profile / sigma_gauss)^2.
# The excess is carried by the SN profile (the BAO profile is ~50x narrower, so the
# joint minimum sits essentially at the BAO minimum and the SN profile contributes
# ~96% of the rise: sn_dchi2_at_joint = 39.9 of 41.7 in the DR1 cell), consistent
# with the SN chi2(fv0) curve railing toward high fv0.
import json
import os

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = WT + "/probes_out/nonparabolicity.json"

out = {"name": "nonparabolicity",
       "definition": "multiplier = dchi2_join / gaussian_sigma^2 = (sigma_profile/sigma_gauss)^2; "
                     "parabolic profiles give exactly 1.0 (cf. the Gaussian Seifert-anchor pairings)",
       "pairings": {}}

# fvsplit.json: the paper's own SN reduction (standard m_b_corr, full stat+sys cov)
# against the committed DR1 BAO+CMB and BAO-only treatments.
fv = json.load(open(WT + "/probes_out/fvsplit.json"))
for bkey, label in [("B1_baocmb_e050", "DR1 BAO+CMB vs SN (fvsplit S1xB1)"),
                    ("B4_baoonly",     "DR1 BAO-only vs SN (fvsplit S1xB4)")]:
    c = fv["tension_matrix"]["S1_standard_fullcov"][bkey]
    out["pairings"][label] = {
        "dchi2_join": c["dchi2_join"],
        "sn_dchi2_at_joint": c["sn_dchi2_at_joint"],
        "sigma_profile": c["sigma"],
        "sigma_gauss": c["gaussian_sigma"],
        "multiplier": round(c["dchi2_join"] / c["gaussian_sigma"] ** 2, 3),
    }

# dr2.json: the DR2 pairings the paper quotes as primary (6.6/4.8 sigma), plus the
# near-Gaussian Seifert-anchor controls.
dr2 = json.load(open(WT + "/probes_out/dr2.json"))
for k, v in dr2["fv_split"].items():
    if "T_profile_shift" not in v:
        continue
    out["pairings"]["dr2.json " + k] = {
        "dchi2_join": v["T_profile_shift"],
        "sigma_profile": v["sigma_profile"],
        "sigma_gauss": v["sigma_gaussian"],
        "multiplier": round(v["T_profile_shift"] / v["sigma_gaussian"] ** 2, 3),
    }

hdr = f"{'pairing':46s} {'dchi2_join':>10s} {'sig_prof':>8s} {'sig_gauss':>9s} {'mult':>6s}"
print(hdr)
for k, v in out["pairings"].items():
    print(f"{k:46s} {v['dchi2_join']:10.3f} {v['sigma_profile']:8.3f} "
          f"{v['sigma_gauss']:9.3f} {v['multiplier']:6.3f}")

m_dr1 = out["pairings"]["DR1 BAO+CMB vs SN (fvsplit S1xB1)"]["multiplier"]
m_dr2 = out["pairings"]["dr2.json dr2_bao_cmb_vs_SN"]["multiplier"]
out["conclusion"] = (f"BAO+CMB-vs-SN multiplier: {m_dr1:.3f} (DR1), {m_dr2:.3f} (DR2) -> "
                     f"quote ~1.9x (one decimal). Supersedes the earlier ~1.8x. "
                     f"Seifert-anchor pairings sit at ~1.0, confirming the excess is the "
                     f"SN profile's non-parabolicity.")
print("\n" + out["conclusion"])

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OUT)
