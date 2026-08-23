"""PALEO Stage-1 inference and event extraction (final, as run in the passed
one-shot test of 2026-08-08). Wraps the trained U-Net: score map -> ranked,
gap-credible, non-overlapping events -> logit-knee truncation -> Stage 2."""
import numpy as np
import torch

from .model import UNet1D, pad_to

DUR_DEFAULT = 3.0 / 24  # evaluation transit duration (days)


def load_model(checkpoint_path, device="cpu"):
    model = UNet1D().to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def nn_score(model, flux, gap, sigma, device="cpu"):
    """Per-cell transit probability on a resampled grid."""
    x = np.stack([flux / sigma, gap]).astype(np.float32)
    with torch.no_grad():
        xt, n = pad_to(torch.from_numpy(x)[None].to(device), 8)
        s = torch.sigmoid(model(xt))[0, :n].cpu().numpy()
    s[gap > 0.5] = 0.0
    return s


def nn_events(score, grid, k=8, min_sep_days=DUR_DEFAULT, live_frac=0.8):
    """Top-k non-overlapping score peaks whose +-half-duration window is mostly
    observed data (a candidate surrounded by gap is not a credible transit).
    Returns (times, scores) in rank order, most significant first."""
    t = grid["t0"] + np.arange(len(score)) * grid["step"]
    live = (grid["gap"] < 0.5).astype(float)
    min_sep = max(1, int(round(min_sep_days / grid["step"])))
    half = max(1, min_sep // 2)
    order = np.argsort(score)[::-1]
    times, scores, taken = [], [], []
    for i in order:
        if score[i] <= 0:
            break
        lo, hi = max(0, i - half), min(len(score), i + half + 1)
        if live[lo:hi].mean() < live_frac:
            continue
        if all(abs(i - j) >= min_sep for j in taken):
            taken.append(i)
            times.append(float(t[i]))
            scores.append(float(score[i]))
            if len(times) >= k:
                break
    return times, scores


def knee_cut_scores(scores, min_k=3, max_k=8, drop=2.0):
    """Cut the ranked event list at the largest logit drop (>= `drop`), the NN
    analogue of the GP pipeline's p-value knee: clean stars self-truncate at
    their true transits; weak-signal stars keep a fuller list."""
    s = np.clip(np.array(scores), 1e-9, 1 - 1e-9)
    l = np.log(s) - np.log(1 - s)
    if len(l) <= min_k:
        return len(l)
    drops = [(l[i] - l[i + 1], i + 1) for i in range(min_k - 1, min(len(l), max_k) - 1)]
    bd, cut = max(drops)
    return cut if bd > drop else min(len(l), max_k)


def detect(model, grid, tol=DUR_DEFAULT, device="cpu", rng=None):
    """Full pipeline on one resampled grid: returns the Stage 2 result dict
    (period, fap, n_matched, ...) plus the extracted events."""
    from .stage2 import stage2_search
    score = nn_score(model, grid["flux"], grid["gap"], grid["sigma"], device)
    times, scores = nn_events(score, grid, min_sep_days=tol)
    ev = times[:knee_cut_scores(scores)] if times else []
    t = grid["t0"] + np.arange(len(grid["flux"])) * grid["step"]
    r = stage2_search(ev, t[0], t[-1], tol=tol,
                      rng=rng or np.random.default_rng(0))
    r["events"] = ev
    r["event_scores"] = scores[:len(ev)]
    return r
