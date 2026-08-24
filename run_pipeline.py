
import argparse

import numpy as np

from paleo.data_utils import resample_uniform
from paleo.detect import (DUR_DEFAULT, knee_cut_scores, load_model, nn_events,
                          nn_score)
from paleo.stage2 import stage2_search


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


def run(tic, sector=None, model_path="models/paleo_unet_v1.pt",
        seed=0, plot=None):
    time, flux, author, sec = fetch_lightcurve(tic, sector)
    print(f"TIC {tic}: {author} sector {sec}, {len(time)} cadences, "
          f"baseline {time[-1]-time[0]:.1f} d")

    grid = resample_uniform(time, flux)
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
          "control / vetting discussion in the docs).")

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
