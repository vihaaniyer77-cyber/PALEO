"""PALEO end-to-end inference on any TESS star -- fully self-contained.

Given a TIC ID: fetch one sector of official-pipeline photometry from MAST,
apply the standard ingestion rules, resample to the 10-minute grid, run the
neural detector, extract events, and run the Stage 2 periodicity search.
Every helper is inlined below; the only external requirements are numpy,
torch, lightkurve, and the frozen checkpoint.

Usage:
    python run_pipeline.py --tic 115591768
    python run_pipeline.py --tic 115591768 --sector 14 --plot out.png
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. MAST ingestion (Technical Reference A.1): official pipelines only,
#    in priority order, each with its systematics-corrected flux column.
#    QLP renamed its corrected column at Sector 56, hence the list.
# ----------------------------------------------------------------------
AUTHORS = ("SPOC", "TESS-SPOC", "QLP")
FLUX_COLUMNS = {
    "SPOC": ["pdcsap_flux"],
    "TESS-SPOC": ["pdcsap_flux"],
    "QLP": ["kspsap_flux", "sys_rm_flux", "det_flux"],
}


def fetch_lightcurve(tic, sector=None):
    """One sector of quality-masked, median-normalised photometry.

    Returns (time_days, flux_normalised, author, sector) with flux as a
    fractional deviation centred on zero -- the convention every downstream
    stage assumes.
    """
    import lightkurve as lk
    search = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    if len(search) == 0:
        raise RuntimeError(f"no TESS light curves at MAST for TIC {tic}")
    for author in AUTHORS:                       # priority order
        rows = search[search.author == author]
        if len(rows) == 0:
            continue
        if sector is not None:
            keep = [i for i in range(len(rows))
                    if f"Sector {sector:02d}" in str(rows.mission[i])
                    or f"Sector {sector}" in str(rows.mission[i])]
            if not keep:
                continue
            rows = rows[keep[0]]
        else:
            rows = rows[0]                       # earliest available sector
        lc = rows.download()
        if lc is None:
            continue
        col = next((c for c in FLUX_COLUMNS[author] if c in lc.colnames), None)
        if col is None:
            continue
        quality = np.asarray(lc["quality"])
        flux = np.asarray(lc[col], float)
        time = np.asarray(lc.time.value, float)
        m = (quality == 0) & np.isfinite(time) & np.isfinite(flux) & (flux > 0)
        time, flux = time[m], flux[m]
        if len(time) < 100:
            continue
        flux = flux / np.median(flux) - 1.0      # fractional, centred on zero
        sec = str(lc.meta.get("SECTOR", "?"))
        return time, flux, author, sec
    raise RuntimeError(f"TIC {tic}: no usable official-pipeline product "
                       f"(SPOC/TESS-SPOC/QLP) with a corrected flux column")


# ----------------------------------------------------------------------
# 2. Resampling to the uniform 10-minute grid + ingestion sanity gate
# ----------------------------------------------------------------------
GRID_MIN = 10.0                       # uniform grid cadence, minutes


def resample_uniform(time, flux, grid_min=GRID_MIN):
    """Resample an irregular light curve onto a uniform grid.

    Returns dict with:
      t0        : grid start time (days)
      flux      : mean flux per cell, 0.0 in gaps  (median-centred input preserved)
      gap       : 1.0 where the cell contains no data, else 0.0
      sigma     : robust per-point scatter of the ORIGINAL flux (for depth conversion)
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    m = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[m], flux[m]
    o = np.argsort(time)
    time, flux = time[o], flux[o]

    step = grid_min / (24 * 60)
    n = int(np.ceil((time[-1] - time[0]) / step)) + 1
    idx = np.clip(((time - time[0]) / step).astype(int), 0, n - 1)

    sums = np.bincount(idx, weights=flux, minlength=n)
    cnts = np.bincount(idx, minlength=n)
    filled = cnts > 0
    out = np.zeros(n, float)
    np.divide(sums, cnts, out=out, where=filled)

    # Coarse-cadence stars (e.g. 30-min QLP) leave empty cells between samples;
    # those are sampling artefacts, not gaps. Interpolate across short holes
    # (<= max_hole_cells) and flag only genuinely unobserved stretches as gaps.
    max_hole_cells = int(np.ceil(45.0 / grid_min))
    cells = np.arange(n)
    gap = np.ones(n, np.float32)
    gap[filled] = 0.0
    if filled.sum() >= 2:
        interp = np.interp(cells, cells[filled], out[filled])
        empty_runs = []
        i = 0
        while i < n:
            if not filled[i]:
                j = i
                while j < n and not filled[j]:
                    j += 1
                empty_runs.append((i, j))
                i = j
            else:
                i += 1
        for a, b in empty_runs:
            if (b - a) <= max_hole_cells and a > 0 and b < n:
                out[a:b] = interp[a:b]
                gap[a:b] = 0.0
    out[gap > 0.5] = 0.0

    live = gap < 0.5
    dif = np.diff(out[live])
    sigma = 1.4826 * np.median(np.abs(dif)) / np.sqrt(2)
    return dict(t0=float(time[0]), flux=out.astype(np.float32),
                gap=gap, sigma=float(max(sigma, 1e-6)), step=step)


def grid_ok(grid, max_mad=0.2, deep=-0.5, max_deep_frac=0.05):
    """Ingestion sanity gate for corrupted photometry. Rejects on robust
    scatter above 20% (most active real star in the benchmark: 13% MAD) or
    when deep cells (< -50%) are WIDESPREAD (> 5% of live cells): corruption
    is pervasive, while a genuine deep eclipse is a narrow feature with a
    small duty cycle -- a quiet star with a -99% dip is a total-eclipse EB
    candidate, not junk, and must pass. Returns (ok, reason)."""
    live = grid["flux"][grid["gap"] < 0.5]
    if len(live) == 0:
        return False, "no live cells"
    mad = float(np.median(np.abs(live - np.median(live))))
    if mad > max_mad:
        return False, f"MAD {mad*100:.1f}% (> {max_mad*100:.0f}%)"
    frac = float((np.abs(live) > -deep).mean())
    if frac > max_deep_frac:
        return False, (f"{frac*100:.1f}% of cells beyond +-{-deep*100:.0f}% "
                       f"(> {max_deep_frac*100:.0f}% -- pervasive scattered-"
                       f"light/systematics, not an eclipse or flare)")
    return True, "ok"


# ----------------------------------------------------------------------
# 3. Stage 1: the neural detector (1-D U-Net, per-cell transit probability)
# ----------------------------------------------------------------------
DUR_DEFAULT = 3.0 / 24  # evaluation transit duration (days)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2),
            nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=k // 2),
            nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    """Input (B, 2, N) [flux/sigma, gap]; output (B, N) logits.
    N must be divisible by 16 (4 pooling levels) - pad_to() handles it."""

    def __init__(self, base=24):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1 = ConvBlock(2, c[0])
        self.enc2 = ConvBlock(c[0], c[1])
        self.enc3 = ConvBlock(c[1], c[2])
        self.enc4 = ConvBlock(c[2], c[3])
        self.pool = nn.MaxPool1d(2)
        self.up3 = nn.ConvTranspose1d(c[3], c[2], 2, stride=2)
        self.dec3 = ConvBlock(c[2] * 2, c[2])
        self.up2 = nn.ConvTranspose1d(c[2], c[1], 2, stride=2)
        self.dec2 = ConvBlock(c[1] * 2, c[1])
        self.up1 = nn.ConvTranspose1d(c[1], c[0], 2, stride=2)
        self.dec1 = ConvBlock(c[0] * 2, c[0])
        self.head = nn.Conv1d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1).squeeze(1)


def pad_to(x, mult=8):
    """Right-pad (B, C, N) or (B, N) to a multiple of `mult`. Returns (padded, n_orig)."""
    n = x.shape[-1]
    pad = (-n) % mult
    if pad:
        x = F.pad(x, (0, pad))
    return x, n


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


# ----------------------------------------------------------------------
# 4. Stage 2: sparse-event periodicity search with permutation FAP
# ----------------------------------------------------------------------
def _phase_cluster_size(times, period, tol):
    """Largest subset of `times` that lands within one tolerance window in phase."""
    ph = np.sort(times % period)
    if len(ph) == 0:
        return 0, 0.0
    ext = np.concatenate([ph, ph + period])
    best, best_phase = 1, float(ph[0])
    j = 0
    for i in range(len(ph)):
        while j < len(ext) and ext[j] <= ext[i] + tol:
            j += 1
        count = j - i
        if count > best:
            best, best_phase = count, float(ext[i] % period)
    return best, best_phase


def candidate_periods(event_times, min_period, max_period, max_divisor=8):
    """Trial periods from pairwise differences and integer divisors thereof."""
    t = np.asarray(sorted(event_times), float)
    cands = set()
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            for k in range(1, max_divisor + 1):
                p = dt / k
                if min_period <= p <= max_period:
                    cands.add(round(p, 6))
    return np.array(sorted(cands))


def score_periods(event_times, periods, tol, baseline, occupancy_min=0.5):
    """Statistic per period: (matches-1) * ln(P/tol) - the log-improbability of
    the observed phase alignment under uniform events. Raw match COUNT is the
    wrong statistic: short trial periods make alignment cheap (tol covers a
    large phase fraction), so junk events beat real signals at small P.

    Occupancy gate: a genuine periodic transit appears in most of its windows;
    a subharmonic + junk coincidence fills only a few of many. Candidates whose
    matched events occupy < occupancy_min of the available windows are vetoed.
    The denominator is capped at the event count: with k extraction slots, a
    short period can fill at most k of its baseline/P windows, so an uncapped
    gate is structurally unpassable at P < k*tol-ish scales (validated
    2026-08-28: capping lifts P<2 d recall 0.24->0.71 exact-period on the dev
    set with control false alarms and P>=3.5 d outcomes unchanged)."""
    n_ev = len(event_times)
    stats = np.full(len(periods), -1.0)
    counts = np.empty(len(periods), int)
    phases = np.empty(len(periods), float)
    for i, p in enumerate(periods):
        m, ph = _phase_cluster_size(event_times, p, tol)
        counts[i] = m
        phases[i] = ph
        n_win = max(1, int(baseline / p))
        if m >= 3 and (m / min(n_win, n_ev)) >= occupancy_min:
            stats[i] = (m - 1) * np.log(max(p / tol, 1.0))
    return stats, counts, phases


def stage2_search(event_times, t_start, t_end, tol,
                  min_period=0.5, max_period=None, n_perm=2000, rng=None):
    """Full Stage 2 search. Returns dict with best period, matched events, FAP."""
    rng = rng or np.random.default_rng(0)
    event_times = np.asarray(sorted(event_times), float)
    n_ev = len(event_times)
    baseline = t_end - t_start
    if max_period is None:
        max_period = baseline / 2.0          # require >=2 cycles
    if n_ev < 3:
        return dict(period=None, score=0.0, n_events=n_ev, n_matched=n_ev,
                    matched_times=[], n_windows=0, fap=1.0, null_median=0.0,
                    note="need >=3 events for a periodicity claim")

    periods = candidate_periods(event_times, min_period, max_period)
    if len(periods) == 0:
        return dict(period=None, score=0.0, n_events=n_ev, n_matched=0,
                    matched_times=[], n_windows=0, fap=1.0, null_median=0.0,
                    note="no candidate periods in range")

    stats, counts, phases = score_periods(event_times, periods, tol, baseline)
    best_idx = int(np.argmax(stats))
    best_stat = float(stats[best_idx])
    if best_stat <= 0:
        return dict(period=None, score=0.0, n_events=n_ev, n_matched=0,
                    matched_times=[], n_windows=0, fap=1.0, null_median=0.0,
                    note="no candidate satisfies the occupancy requirement")
    best_period = float(periods[best_idx])
    best_phase = float(phases[best_idx])

    ph = event_times % best_period
    matched = ((ph - best_phase) % best_period) <= tol
    n_windows = int(baseline / best_period) + 1

    # permutation null: uniform event times, same count, same search
    null_best = np.empty(n_perm, float)
    for b in range(n_perm):
        fake = np.sort(rng.uniform(t_start, t_end, n_ev))
        fp = candidate_periods(fake, min_period, max_period)
        if len(fp) == 0:
            null_best[b] = 0.0
            continue
        fs, _, _ = score_periods(fake, fp, tol, baseline)
        null_best[b] = max(fs.max(), 0.0)
    fap = float((null_best >= best_stat).mean())

    return dict(period=best_period, epoch_phase=best_phase, score=best_stat,
                n_events=n_ev, n_matched=int(matched.sum()),
                matched_times=event_times[matched].tolist(),
                n_windows=n_windows, fap=fap,
                null_median=float(np.median(null_best)))


# ----------------------------------------------------------------------
# 5. The pipeline
# ----------------------------------------------------------------------
def run(tic, sector=None, model_path="models/paleo_unet_v1.pt",
        seed=0, plot=None):
    time, flux, author, sec = fetch_lightcurve(tic, sector)
    print(f"TIC {tic}: {author} sector {sec}, {len(time)} cadences, "
          f"baseline {time[-1]-time[0]:.1f} d")

    grid = resample_uniform(time, flux)
    ok, reason = grid_ok(grid)
    if not ok:
        raise RuntimeError(f"TIC {tic}: light curve fails the ingestion "
                           f"sanity gate ({reason}) -- not a usable product")
    print(f"grid: {len(grid['flux'])} cells | sigma = {grid['sigma']*100:.4f}% "
          f"| live fraction {(grid['gap']<0.5).mean():.2f}")

    model = load_model(model_path, "cpu")
    score = nn_score(model, grid["flux"], grid["gap"], grid["sigma"])
    times, scores = nn_events(score, grid)
    cut = knee_cut_scores(scores) if times else 0
    events = times[:cut]
    print(f"events: {len(times)} extracted, knee kept {cut} "
          f"(scores {[round(s,3) for s in scores[:cut]]})")

    t = grid["t0"] + np.arange(len(grid["flux"])) * grid["step"]
    r = stage2_search(events, t[0], t[-1], tol=DUR_DEFAULT,
                      rng=np.random.default_rng(seed))
    if r["period"] is None:
        print(f"STAGE 2: abstained ({r.get('note','')})")
    else:
        tier = ("DETECTION (FAP < 0.05)" if r["fap"] < 0.05 else
                "candidate tier (0.05-0.15)" if r["fap"] < 0.15 else
                "not significant")
        print(f"STAGE 2: period = {r['period']:.4f} d | "
              f"matched {r['n_matched']}/{r['n_events']} events | "
              f"FAP = {r['fap']:.4f} -> {tier}")
    print("note: a periodic signal here is 'periodic dips', not a confirmed "
          "planet -- eclipsing binaries pass this test too (see paired-"
          "control / vetting discussion in the docs). Claims with P < 2 d "
          "should additionally be checked against the star's rotation period.")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fp = grid["flux"].copy() * 100
        fp[grid["gap"] > 0.5] = np.nan
        sp = score.copy()
        sp[grid["gap"] > 0.5] = np.nan
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
        a1.plot(t, fp, lw=0.6)
        a1.set_ylabel("relative flux (%)")
        title = f"TIC {tic} ({author} s{sec})"
        if r["period"]:
            title += f" | P = {r['period']:.3f} d, FAP = {r['fap']:.4f}"
        a1.set_title(title)
        a2.plot(t, sp, lw=0.8, color="tab:purple")
        for ev in events:
            a2.axvline(ev, color="tab:red", lw=0.8, alpha=0.6)
        a2.set_ylim(-0.05, 1.05)
        a2.set_xlabel("time (BTJD, days)")
        a2.set_ylabel("transit prob.")
        fig.tight_layout()
        fig.savefig(plot, dpi=130)
        print(f"figure saved: {plot}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--sector", type=int, default=None,
                    help="TESS sector (default: earliest available)")
    ap.add_argument("--model", default="models/paleo_unet_v1.pt")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the permutation FAP")
    ap.add_argument("--plot", default=None, help="save a figure to this path")
    a = ap.parse_args()
    run(a.tic, a.sector, a.model, a.seed, a.plot)
