"""Stage 2: sparse-event periodicity search.

Input:  candidate event mid-times from Stage 1 (some true transits, some junk).
Output: best trial period, the subset of events consistent with it, and an
        empirical (permutation) false-alarm probability.

Design:
  - Trial periods come from pairwise event-time differences and their integer
    divisors (a true period P must appear as dt/k for some event pair).
  - Score(P) = the largest number of events that phase-align within a tolerance
    window (tolerance = transit duration by default).
  - Significance: permutation null. Junk events are ~uniform over the observing
    window, so we resample event times uniformly (same count) many times and ask
    how often the best achievable score >= the observed one.
"""
import numpy as np


def _phase_cluster_size(times, period, tol):
    """Largest subset of `times` that lands within one tolerance window in phase."""
    ph = np.sort(times % period)
    if len(ph) == 0:
        return 0, 0.0
    # circular: duplicate shifted by period to handle wraparound
    ext = np.concatenate([ph, ph + period])
    best, best_phase = 1, float(ph[0])
    j = 0
    for i in range(len(ph)):
        # widen window [ext[i], ext[i]+tol]
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
    matched events occupy < occupancy_min of the available windows are vetoed."""
    stats = np.full(len(periods), -1.0)
    counts = np.empty(len(periods), int)
    phases = np.empty(len(periods), float)
    for i, p in enumerate(periods):
        m, ph = _phase_cluster_size(event_times, p, tol)
        counts[i] = m
        phases[i] = ph
        n_win = max(1, int(baseline / p))
        if m >= 3 and (m / n_win) >= occupancy_min:
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

    # which events matched
    ph = event_times % best_period
    lo = best_phase
    matched = ((ph - lo) % best_period) <= tol

    # expected number of windows for sanity (transits in baseline)
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
