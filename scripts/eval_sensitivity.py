"""Dev-set sensitivity and duration curves (notebook 11 Cells 2-3, final results
2026-08-08). Protocol identical to the GP prototype's notebook-05 curve: single
3 h box transit per star, epoch at a gap-safe point, recovery = any top-3
ranked event within one duration of the epoch.

Measured (30 dev stars):
  depth   0.2%: 33%   0.5%: 57%   1%: 73%   2%: 87%   control: 0/30
  (GP:          11%         56%       67%       89%             0/9)
  duration @1%   1.5h: 63%   3h: 73%   6h: 77%     (GP: 33% / 67% / 67%)

Usage: python eval_sensitivity.py --base /path/to/PALEO_colab
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paleo.detect import load_model, nn_score, nn_events

DUR = 3.0 / 24


def load_grids(base, folder):
    out = []
    for fn in sorted(os.listdir(f"{base}/data/{folder}")):
        z = np.load(f"{base}/data/{folder}/{fn}")
        out.append(dict(flux=z["flux"], gap=z["gap"], sigma=float(z["sigma"]),
                        t0=float(z["t0"]), step=float(z["step"]), tic=fn))
    return out


def safe_epoch(g, dur=DUR, frac=0.35):
    """A live cell ~frac through the observed span whose +-2 durations are fully
    live. (Naive mid-grid placement lands in the TESS downlink gap; mid-observed
    placement lands at the gap edge where systematics concentrate.)"""
    t = g["t0"] + np.arange(len(g["flux"])) * g["step"]
    live = g["gap"] < 0.5
    w = int(round(2 * dur / g["step"]))
    ok = [i for i in np.where(live)[0][w:-w] if live[i - w:i + w + 1].all()]
    return t[ok[int(frac * len(ok))]] if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(f"{args.base}/models/paleo_unet_v1.pt", device)
    dev_grids = load_grids(args.base, "dev_grids")

    rows = []
    for g in dev_grids:
        t = g["t0"] + np.arange(len(g["flux"])) * g["step"]
        ep = safe_epoch(g)
        if ep is None:
            continue
        for depth in (0.02, 0.01, 0.005, 0.002, 0.0):
            flux = g["flux"].copy()
            if depth > 0:
                m_in = (np.abs(t - ep) < DUR / 2) & (g["gap"] < 0.5)
                if m_in.sum() < 2:
                    continue
                flux[m_in] -= depth
            score = nn_score(model, flux, g["gap"], g["sigma"], device)
            times, _ = nn_events(score, g, k=8)
            rows.append(dict(tic=g["tic"], depth=depth,
                             recovered=any(abs(e - ep) < DUR for e in times[:3])))
    df = pd.DataFrame(rows)
    print(df.groupby("depth").recovered.agg(["mean", "count"]).round(2))

    rows = []
    for dur_h in (1.5, 6.0):
        dur = dur_h / 24
        for g in dev_grids:
            t = g["t0"] + np.arange(len(g["flux"])) * g["step"]
            ep = safe_epoch(g, dur)
            if ep is None:
                continue
            flux = g["flux"].copy()
            m_in = (np.abs(t - ep) < dur / 2) & (g["gap"] < 0.5)
            if m_in.sum() < 2:
                continue
            flux[m_in] -= 0.01
            score = nn_score(model, flux, g["gap"], g["sigma"], device)
            times, _ = nn_events(score, g, k=8, min_sep_days=dur)
            rows.append(dict(dur_h=dur_h,
                             recovered=any(abs(e - ep) < dur for e in times[:3])))
    print(pd.DataFrame(rows).groupby("dur_h").recovered.agg(["mean", "count"]).round(2))


if __name__ == "__main__":
    main()
