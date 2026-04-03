# PALEO — Physics-Aware Light-curve Exoplanet Observer

> *Teaching machines to find worlds.*

PALEO is an end-to-end exoplanet transit detection pipeline that combines a **Temporal Convolutional Network (TCN)** with a **Physics-Aware Validator (PAV)** to detect and validate planetary signals in Kepler and TESS photometry — automatically, at scale, and with near-human precision.

---

## The Problem

Space telescopes like Kepler and TESS produce millions of light curves. Planetary transits — the tiny dips in brightness as a planet passes in front of its star — are buried in noise, stellar variability, and instrumental artefacts. Classical pipelines either miss real signals or flood astronomers with false positives (eclipsing binaries, pulsating stars, noise spikes) that must be reviewed by hand. This bottleneck is the limiting factor in modern exoplanet discovery.

---

## The Solution

PALEO operates in two stages:

### Stage 1 — TCN Proposer
A deep **Temporal Convolutional Network** trained on synthetic and real light curves scans each 5-day window of photometry and produces a per-cadence probability heatmap, flagging intervals that contain a transit-shaped event. The TCN:
- Operates on raw, detrended flux — no phase-folding required
- Uses dilated convolutions with receptive fields spanning the full 5-day window
- Outputs separate channels for transits and stellar flares
- Runs on GPU via PyTorch in seconds per sector

### Stage 2 — Physics-Aware Validator (PAV)
Every TCN candidate passes through five sequential astrophysical gates:

| Gate | Test | What it catches |
|------|------|-----------------|
| 1 | Lomb-Scargle pulsator check (4–48 cpd) | Delta Scuti pulsating stars |
| 2 | Structural: ≥3 TCN peaks required | Noise spikes, single events |
| 3 | BLS SDE > 6.0 on real flux | Non-periodic artefacts |
| 4 | EB rejection: depth, secondary eclipse, odd/even asymmetry | Eclipsing binaries |
| 5 | Batman χ² shape fit (limb-darkened model) | Remaining morphological impostors |

Only candidates that survive all five gates are promoted to planet candidates.

---

## Results

Benchmarked on **10,000 synthetic light curves** covering four planet classes (Hot Jupiter, Saturn, Neptune, Super-Earth) and four false positive populations:

| Metric | Value |
|--------|-------|
| **Precision** | 94.88% |
| **Recall** | 98.15% |
| **F1-Score** | **96.49%** |
| Pulsator FP rate | **0%** |
| Noise spike FP rate | **0%** |

---

## Beyond Standard Detection

PALEO's geometry-based validation extends naturally to scenarios where classical period-folding methods fail:

- **Circumbinary planets** — shape-only classification (batman χ² + in-transit flatness) identifies CBP transits with no period information
- **Oscillating red giants** — the 5-day rolling-median detrend preserves 94.7% of solar-like oscillation power, and the pulsator gate does not reject red giants (νmax < 4 cpd)

---

## Architecture

```
Raw TESS/Kepler LC
        │
   Detrend (rolling median, 5-day window)
        │
   TCN Inference (PyTorch, GPU)
        │
   Peak Detection (scipy.signal.find_peaks)
        │
   ┌────▼──────────────────────────────┐
   │     Physics-Aware Validator       │
   │  Gate 1: Pulsator (LS power)      │
   │  Gate 2: ≥3 peaks structural      │
   │  Gate 3: BLS SDE > 6.0            │
   │  Gate 4: EB 3-gate rejection      │
   │  Gate 5: Batman χ² shape fit      │
   └────────────────────────────────┬──┘
                                    │
                          Planet Candidates
```

---


## Quick Start

```bash
pip install lightkurve astropy batman-package scipy torch tqdm
```

```python
# Evaluate on 10,000 synthetic light curves
python evaluate_pipeline_fast.py

# Run on a TESS sector
python tess_pipeline_updated.py
```

Requires `tcn_epoch_020.pt` model weights in the working directory (or `/content/` in Colab).

---

## Dependencies

- `torch` ≥ 2.0
- `lightkurve` ≥ 2.4
- `astropy` ≥ 5.0
- `batman-package`
- `scipy`, `numpy`, `matplotlib`, `tqdm`

---

## Next Steps

- [ ] Full TESS sector survey (Sectors 1–26)
- [ ] Multi-sector stacking for Super-Earth sensitivity
- [ ] Circumbinary system searches

---


---

*Built with PyTorch, batman, and a lot of synthetic transits.*
