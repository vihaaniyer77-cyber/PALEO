"""PALEO neural-network data utilities: resampling, injection, label generation.

Written by notebook 09, imported by notebooks 10-12.
"""
import numpy as np

GRID_MIN = 10.0                       # uniform grid cadence, minutes
GRID_DAY = GRID_MIN / (24 * 60)


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


def transit_profile(dt, depth, duration, ingress_frac):
    """Trapezoidal transit: dt = time from mid-transit (days). Returns flux deficit >= 0."""
    half = duration / 2.0
    t_flat = half * (1.0 - ingress_frac)
    prof = np.zeros_like(dt, float)
    a = np.abs(dt)
    in_flat = a <= t_flat
    in_ramp = (a > t_flat) & (a < half)
    prof[in_flat] = depth
    denom = max(half - t_flat, 1e-9)
    prof[in_ramp] = depth * (half - a[in_ramp]) / denom
    return prof


def inject_periodic(grid, period, epoch_frac, depth, duration, ingress_frac, shape="trap"):
    """Inject a periodic transit into a resampled grid. Returns (flux, mask, n_true).

    mask: 1.0 where |t - transit centre| < duration/2 AND the cell is not a gap.
    n_true counts transits with >= 2 non-gap in-transit cells (matches the GP harness rule).
    """
    n = len(grid["flux"])
    t = grid["t0"] + np.arange(n) * grid["step"]
    flux = grid["flux"].copy()
    mask = np.zeros(n, np.float32)

    e0 = t[0] + epoch_frac * period
    k, n_true = 0, 0
    while e0 + k * period < t[-1]:
        tc = e0 + k * period
        dt = t - tc
        if shape == "box":
            deficit = np.where(np.abs(dt) < duration / 2, depth, 0.0)
        else:
            deficit = transit_profile(dt, depth, duration, ingress_frac)
        live = grid["gap"] < 0.5
        flux[live] -= deficit[live].astype(np.float32)
        m_in = (np.abs(dt) < duration / 2) & live
        mask[m_in] = 1.0
        if m_in.sum() >= 2:
            n_true += 1
        k += 1
    return flux, mask, n_true


def make_training_sample(grid, rng, p_inject=0.8,
                         depth_range=(0.002, 0.03), dur_range_h=(1.0, 6.0),
                         period_range=(2.0, 10.0)):
    """One training sample from a resampled star.

    Returns x (2, n) float32 [flux / sigma, gap], y (n,) float32 mask, meta dict.
    """
    if rng.random() < p_inject:
        depth = float(np.exp(rng.uniform(np.log(depth_range[0]), np.log(depth_range[1]))))
        dur = float(rng.uniform(*dur_range_h)) / 24.0
        period = float(rng.uniform(*period_range))
        ef = float(rng.uniform(0.05, 0.95))
        fi = float(rng.uniform(0.1, 0.5))
        flux, mask, n_true = inject_periodic(grid, period, ef, depth, dur, fi)
        meta = dict(injected=True, depth=depth, duration_h=dur * 24,
                    period=period, epoch_frac=ef, n_true=n_true)
    else:
        flux, mask = grid["flux"].copy(), np.zeros_like(grid["flux"])
        meta = dict(injected=False, depth=0.0, duration_h=0.0, period=0.0,
                    epoch_frac=0.0, n_true=0)
    x = np.stack([flux / grid["sigma"], grid["gap"]]).astype(np.float32)
    return x, mask.astype(np.float32), meta
