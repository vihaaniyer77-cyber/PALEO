# PALEO Results Compendium

Every dataset, every experiment, every number — one document, in order.
Raw per-trial data: `PALEO_colab/results/` on Drive (CSV names given per test).

---

## 1. Datasets

**List A — 15 stars** (period-stratified; `list_a_rotation_validation.csv`).
Source: TARS catalog (Boyle et al. 2026; 1,046,317 stars → 702,694 after
single/non-binary/single-period cuts). 3 stars per rotation-period bucket
[0–0.5, 0.5–1, 1–4, 4–8, 8–30 d]; official pipelines only (SPOC/TESS-SPOC/QLP),
corrected flux columns, quality==0. Cool dwarfs, Teff 3,060–5,690 K, Tmag
10.7–15.7; single sectors 23–28 d (one 51 d). The 9 stars with P_rot > 1 d are
the "slow rotators" used in all injection tests: TICs 193796923, 417556024,
57097771, 69155627, 115921321, 460384642, 369580651, 274211455, 427450978.

**List B — 15 stars** (activity-stratified; `list_b_activity_stratified.csv`).
Same TARS cuts, List A excluded, official-pipeline availability required.
Activity = MAD of normalised flux, spanning 0.06% → 13.1% (~200×), split into
quiet/moderate/active tertiles. The 11 with P_rot > 1 d joined the benchmark:
258619528, 27548079, 116609821, 1206084, 357896105, 88862679, 292854663,
212289253, 128487216, 448936195, 46999652 (excluded fast rotators: 317069561,
23439606, 48442386, 103292020). Zero role in any calibration.

**List C — 15 NASA-confirmed TOI planets on 10 stars**
(`list_c_toi_case_study.csv`): TOIs 1052, 1054, 1055, 1062, 1064 (×2), 1073,
1075, 1130 (×2), 1135, 1136 (×4). Bright (Tmag 8.1–13.7), single SPOC sectors.
Used only for the blind case study.

**NN training/dev — 330 + 30 stars** (`data/train_grids`, `data/dev_grids`).
Fresh TARS draw (seed 20260808) from a 510,861-star pool with every List A/B
star quarantined; P_rot > 1 d; resampled to uniform 10-min grids. Train = 330,
dev = 30 (iteration allowed), List A/B = final test only (one shot).

**The canonical benchmark set** = List A slow rotators (9) + List B slow
rotators (11) = 20 stars. Injections pre-registered (seed 20260807): one
1%-deep, 3-h box transit train per star, P ~ U(3,9) d, epoch ~ U(0.1,0.9)·P,
plus one untouched control run per star. Identical for GP, CETRA, NN.

---

## 2. Component tests (GP-prototype era)

**T1. Detection-threshold calibration** (synthetic, no-signal GP light curves).
Default num_sigma_threshold=3 flagged 30–40% of signal-free data; cause:
correlated GP residuals (measured 99th percentile 3.92σ vs Gaussian 2.58σ).
At threshold 6: 1–3% flagging across 5 seed/period combinations; identical for
thresholds 5–8. Adopted: 6.

**T2. Rotation-period estimator** (all 15 List A stars vs TARS truth;
`period_validation_results.csv`). Nahiku get_dominant_period(): median error
46.0%, 0/15 within 10% (defects: BTJD-as-period-cap fallback; uniform-sampling
periodogram; amplitude-dependent prominence). Replacement (bounded LS + Baluev
FAP<1e-3 gate): median error 3.7%; 12/15 accepted; of accepted, 8/12 within
10%, 10/12 within 10% allowing factor-2 harmonic; 3 correctly declined
(verified genuine non-detections by phase-fold). Side finding: TIC 57097771's
TARS period (2.634 d) is wrong; true 0.322 d (clean fold, ~25× peak).

**T3. Data-preparation A/B tests.** (a) Community pipelines
(eleanor/CDIPS) caused every catastrophic period failure in a mixed sample →
official pipelines only. (b) lightkurve serves raw sap_flux for QLP: TIC
69642585 est. 6.98 d raw vs 0.184 d (exact) with kspsap_flux. (c) Quality
masking: TIC 436403069 0.835 d unmasked vs 0.329 d (exact) masked; unmasked,
no-signal TIC 460384642 falsely passed the FAP gate.

**T4. Balmung prewhitening calibration** (48 trials: 3-h/1% box at 6 epochs ×
9 List-A slow rotators; `balmung_depth_preservation.csv`,
`balmung_signal_removal.csv`):

| minimum_snr | sinusoids removed | variance removed | depth kept (median) | trials losing >half |
|---|---|---|---|---|
| 0 (default) | 200 | 94.5% | 8.0% | 97.9% |
| 2 | 36 | 66.0% | 50.2% | 47.9% |
| 3 | 8 | 48.5% | 84.9% | 12.5% |
| **4 (adopted)** | 4 | 43.0% | **98.4%** | **0%** |
| 5 | 3 | 40.6% | 99.0% | 0% |
| 10 | 2 | 30.4% | 100% | 0% |

**T5. Search-mode comparison** (real List A stars). GreedySearch:
mll/nlpd metrics crash (NotPSDError — posterior evaluated at own training
points); rmse/msll never terminate expansion; flagged fraction vs binning
0.4%→52.9% on one star (chaotic). ExhaustiveSearch (n=1000, top-3): 1% depth
recovered 2/3 at 1.0–1.2% flagged; 0.2% 0/3; control 0/3 at 1.2%. Adopted:
ExhaustiveSearch. Its parametric p-values are anti-conservative (Bonferroni
0.05 flags 20–68 "events" on controls) → rankings only.

---

## 3. Stage-1 sensitivity characterisation

**T6. GP curve** (9 List-A slow rotators, 36-min bins, single gap-safe 3-h box,
top-3 interval flagging; `stage1_sensitivity_sweep.csv`):

| depth | recovered | fraction |
|---|---|---|
| control | 0/9 | 0.000 |
| 0.2% | 1/9 | 0.111 |
| 0.5% | 5/9 | 0.556 |
| 1% | 6/9 | 0.667 |
| 2% | 8/9 | 0.889 |

Duration axis at 1%: 1.5 h 3/9 (33%), 3 h 6/9, 6 h 6/9. Interval-grid
coarseness (5.8× fewer intervals): identical recovery, runtime 202→~90 s/star.
GPU vs CPU: ~1.5× at best (small matrices); not the lever.
(n=9: binomial errors ≈ ±15 pp per point.)

**T7. NN curve** (30 dev stars, 10-min grid, same protocol;
`nn_sensitivity_dev.csv`):

| depth | NN | GP reference |
|---|---|---|
| control | 0/30 | 0/9 |
| 0.2% | 0.33 | 0.11 |
| 0.5% | 0.57 | 0.56 |
| 1% | 0.73 | 0.67 |
| 2% | 0.87 | 0.89 |

Duration at 1%: 1.5 h **0.63** (GP 0.33), 3 h 0.73, 6 h 0.77 (GP 0.67).
Supporting: learning curve dev-loss 0.787/0.615/0.542/0.547 at 50/100/200/330
training stars (saturates ~200; `nn_learning_curve.csv`); training best dev
0.5553 @ step 1200 (`nn_training_history.csv`).

---

## 4. End-to-end runs (full pipeline → Stage 2 → FAP)

**T8. GP, tuned set** (9 List-A stars, P 3.3–8.4 d, 1%/3 h + 9 controls;
`e2e_results.csv` v1, `e2e_results_v2.csv` final pipeline). v1: 4/9 confirmed
(417556024 · 115921321 · 369580651 · 274211455). v2: 3/9 confirmed + 1
candidate (69155627, .084) + 1 correct-not-significant (369580651, .162).
Controls 0/9 both. Extraction ablation verdict: v2 kept (mechanism), stars
declared tuned → fresh set required.

**T9. GP, pre-registered fresh set** (canonical 20+20; `fresh_validation.csv`).
Confirmed 5/20: 417556024 (4.437/.002), 115921321 (3.880/.0000), 274211455
(5.126/.001), 116609821 (5.502/.001), 46999652 (8.500/.012). Near: 292854663
(4.031/.053); harmonic: 1206084 (6.313 = 2×3.152, .026). Unwinnable (<3
transits landed): 69155627, 460384642, 27548079, 128487216, 448936195 — all
correctly abstained. Excluded by paired control: 88862679.
**Totals: 5/20 overall; 5/15 winnable (33.3%); controls 0/20 @ FAP<.05, 3/20
in 0.05–0.15 tier. Clean subset (n=16): R 0.3125, P 1.00, F1 0.48.**

**T10. CETRA head-to-head** (same 20+20, full native cadence, SNR≥7.1;
`cetra_comparison.csv`). Recovered 13/20 (65%); 14/20 with factor-2 harmonic.
Controls above threshold: 7/20 (spurious SNRs incl. 11.0, 24.3, 69.4, 206.6 vs
genuine 10.2–297.8 — distributions fully overlap; no separating threshold).
**Clean subset (n=16): tp 13, fp 5 → R 0.8125, P 0.722, F1 0.76.** Baseline
validation (T0): TOI-1052.01 recovered at 9.1447 d vs 9.1398 (0.05%), SNR 9.9,
1.4 s.

**T11. NN, dev end-to-end** (30 dev stars + 30 controls, seed 20260811;
`nn_e2e_dev.csv`). Winnable 24/30; confirmed 16/24 (66.7%); controls 2/30 @
.05 (both intrinsic signals, identified by paired controls), 10/30 in tier
(4 intrinsic + 6 chance-consistent). Post-exclusion: 16/20 = 80%.

**T12. NN, one-shot final test** (canonical 20+20, frozen model;
`final_test_nn.csv`). Confirmed 11/20: 193796923 (4.628/.004), 417556024
(4.427/.002), 115921321 (3.885/.000), 369580651 (5.014/.002), 274211455
(5.132/.001), 258619528 (7.743/.046), 116609821 (5.517/.001), 1206084
(3.167/.006 — fundamental, not harmonic), 357896105 (5.333/.003), 448936195
(7.663/.017), 46999652 (8.493/.013). Near: 57097771 (6.257/.080, 0.5% err);
292854663 (4.049/.169). Paired-control exclusions: 460384642, 427450978 (EB,
21.66 d in both twins), 212289253, 128487216.
**Pre-registered criteria: recall ≥10/20 → 11/20 MET; controls ≤1/16 @.05 →
0/16 MET; 0/16 @.01 → 0/16 MET. Clean winnable 10/13 (76.9%). Clean subset:
R 0.6875, P 1.00, F1 0.81.**

**Three-way (clean subset, n=16):**

| | recall | precision | F1 | control FAs |
|---|---|---|---|---|
| GP prototype | 0.31 | 1.00 | 0.48 | 0 |
| CETRA (SNR≥7.1) | 0.81 | 0.72 | 0.76 | 5/16 |
| NN + Stage 2 (FAP<.05) | 0.69 | 1.00 | **0.81** | **0/16** |

Recall and F1 differences vs CETRA: not significant at n=16. False-alarm
difference (5/16 vs 0/16): the significant result. At ~1% survey prevalence,
FPR 0.31 → candidate-list precision ≈2%; calibrated FAP ≈10%+.

---

## 5. Blind case study — real planets (T13; `toi_recovery.csv`, gallery
`toi_fold_gallery_v2.png`)

Frozen NN pipeline, no injections, single SPOC sectors, List C.
Recovered at FAP<0.05 (5 stars): TOI-1073 (3.931 d, 0.2% err, depth 31,511
ppm) · TOI-1130c (4.076, 0.19%, 2,650 ppm; multi-planet system) · TOI-1135
(8.083, 0.69%, 5,705 ppm, FAP .0000) · TOI-1052 (9.208, 0.75%, **358 ppm**) ·
TOI-1055 (8.753 = P/2 of 17.47 d, 0.2%, 1,200 ppm, from 2 transits).
Subharmonic detection: TOI-1054 (3.847 ≈ P/4 of 15.51 d, 513 ppm, FAP .012).
Missed — below per-event SNR: TOI-1062 (487 ppm), TOI-1064 b/c (1,122/1,206
ppm), TOI-1136 (813 ppm; its 2,320 ppm sibling had 2 transits). Missed by
design: TOI-1075 (P=0.60 d, outside sparse-event regime).
**Every outcome explained by per-event SNR ≳ 4–5 (star-relative, from
σ-normalised inputs) plus the ≥3-transit rule.**

---

## 6. Negative result (T14; dev-only)

v2 model (depths extended to 0.05% + detectability-weighted loss;
`models/paleo_unet_v2_deep.pt`). Targeted paired gate, 400 samples at SNR 2–6:
top-k overlap v1 vs v2 = .039/.053 (SNR 2–3), .073/.086 (3–4), .136/.144
(4–5), .179/.167 (5–6) — all within noise. Conclusion: single-event floor
~SNR 4–5 robust to loss weighting; v1 remains canonical; depth path is
Stage 2.5 stacking + multi-sector, not detector retraining.

---

## 7. Validity domain (applies to all of the above)

Single-sector (~25 d) TESS photometry; cool dwarfs (K/M); rotation periods
> 1 d (fast rotators untested); injected transits box/trapezoid, 0.2–3%
depth, 1–6 h duration, P = 2–10 d; official-pipeline flux only. Untested:
fast rotators, realistic limb-darkened shapes, multi-sector baselines, other
spectral types.
