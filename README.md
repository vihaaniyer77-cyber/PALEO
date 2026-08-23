# PALEO — Period-Agnostic Transit Detection

Exoplanet transits are detected **first, with no period assumption** (Stage 1,
a learned per-cadence detector), and orbital periods are recovered **after**,
from the sparse event list, with calibrated significance (Stage 2,
permutation FAP). Headline result: on a one-shot pre-registered test the
pipeline matched a classical period-first search's recall at **zero control
false alarms** (11/20 strict recall, 0/16 controls at FAP < 0.05), and
blind-recovered a confirmed 358 ppm planet.

## Layout

    paleo/                       the pipeline package
      data_utils.py                10-min resampling, gap channel, sigma,
                                   transit injection, training-sample factory
      model.py                     UNet1D (~1.0M params) + masked weighted BCE
      detect.py                    inference: score map -> ranked events ->
                                   knee cut
      stage2.py                    sparse-event period search + permutation FAP
    scripts/
      select_training_stars.py     TARS pool + evaluation-star quarantine +
                                   fixed-seed candidate draw
      download_grids.py            resumable MAST downloader -> training grids
      train.py                     training loop (early stopping, resumable)
      eval_sensitivity.py          depth/duration sensitivity curves
      final_test.py                the pre-registered one-shot final test
    run_pipeline.py              end-to-end inference on any star:
                                   python run_pipeline.py --tic 115591768
    tests/
      test_injected_recovery.py    full-chain recovery of a 1%/3h injection
      test_control_quiet.py        same star, no injection -> must stay quiet

## Quick start

    pip install numpy torch matplotlib pandas lightkurve
    python run_pipeline.py --tic <TICID> --plot out.png

`run_pipeline.py` fetches one sector of official-pipeline TESS photometry
(SPOC / TESS-SPOC / QLP priority, quality-masked, corrected flux columns),
resamples it, runs the frozen detector, and reports Stage 2's period and
false-alarm probability. A detection means "significant periodic dips" —
eclipsing binaries pass too; vetting is separate.

## Tests

    PALEO_GRID=<path to TIC_272566345.npz> \
    PALEO_MODEL=<path to paleo_unet_v1.pt> \
    python tests/test_injected_recovery.py && python tests/test_control_quiet.py

Both tests run the real pipeline on a real spotted slow rotator (5.6% spot
amplitude): one asserts an injected 1%/3 h box train is recovered at the
right period with FAP < 0.05; the other asserts the untouched light curve
produces no claim. Together they check detection *and* silence — a detector
is only useful if it does both.

## Data and weights

Light-curve grids and the frozen checkpoint (`paleo_unet_v1.pt`, seed 42,
best dev loss 0.5553) are not in the repo. Grids are rebuildable with
`scripts/select_training_stars.py` + `scripts/download_grids.py` (TARS
catalog: zenodo.org/records/19917941). Reproduction seeds: 20260808
(training-star selection), 20260807 (final-test injections), 42/999
(training/dev eval).

## Validity domain

Single-sector TESS photometry; K/M slow rotators (P_rot > 1 d); transit
depths 0.2–3%, durations 1–6 h, periods 2–10 d; official pipelines only.
Fast rotators, multi-sector baselines, and limb-darkened shapes are
documented as untested or out of domain.
