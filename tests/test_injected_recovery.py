"""Test 1: end-to-end injected-transit recovery on a real slow rotator.

Injects the standard evaluation signal (1% / 3 h box train, P = 5.2 d) into
TIC 272566345 (dev star: P_rot ~ 4.5 d, 5.6% spot amplitude, sigma = 0.031%)
and asserts the full pipeline recovers it: knee keeps exactly the observable
transits, Stage 2 returns the injected period within 5% at FAP < 0.05.

Requires local data + checkpoint (not in the repo):
    PALEO_GRID  = path to TIC_272566345.npz        (dev grid)
    PALEO_MODEL = path to paleo_unet_v1.pt

Run:  python tests/test_injected_recovery.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paleo.data_utils import inject_periodic                      # noqa: E402
from paleo.detect import (DUR_DEFAULT, knee_cut_scores, load_model,  # noqa: E402
                          nn_events, nn_score)
from paleo.stage2 import stage2_search                            # noqa: E402

GRID = os.environ.get("PALEO_GRID", "data/dev_grids/TIC_272566345.npz")
MODEL = os.environ.get("PALEO_MODEL", "models/paleo_unet_v1.pt")
P_INJ, EF, DEPTH = 5.2, 0.3, 0.01


def main():
    d = np.load(GRID)
    grid = dict(t0=float(d["t0"]), flux=d["flux"].astype(float),
                gap=d["gap"].astype(float), sigma=float(d["sigma"]),
                step=float(d["step"]))
    t = grid["t0"] + np.arange(len(grid["flux"])) * grid["step"]

    flux_inj, mask, n_true = inject_periodic(grid, P_INJ, EF, DEPTH,
                                             DUR_DEFAULT, 0.0, shape="box")
    assert n_true >= 3, f"injection landed only {n_true} observable transits"

    model = load_model(MODEL, "cpu")
    score = nn_score(model, flux_inj, grid["gap"], grid["sigma"])
    times, scores = nn_events(score, grid)
    cut = knee_cut_scores(scores)
    events = times[:cut]

    # every kept event must sit on an injected transit
    centers = []
    tc = t[0] + EF * P_INJ
    while tc < t[-1]:
        centers.append(tc)
        tc += P_INJ
    for ev in events:
        dist = min(abs(ev - c) for c in centers)
        assert dist < DUR_DEFAULT, f"kept event at {ev:.3f} is junk ({dist:.2f} d off)"

    r = stage2_search(events, t[0], t[-1], tol=DUR_DEFAULT,
                      rng=np.random.default_rng(0))
    assert r["period"] is not None, "Stage 2 abstained on a 32-sigma injection"
    err = abs(r["period"] - P_INJ) / P_INJ
    assert err < 0.05, f"period {r['period']:.4f} vs {P_INJ} ({err*100:.1f}% off)"
    assert r["fap"] < 0.05, f"FAP {r['fap']:.4f} not significant"

    print(f"PASS: kept {cut} events (all true) | period {r['period']:.4f} d "
          f"(err {err*100:.2f}%) | matched {r['n_matched']}/{r['n_events']} "
          f"| FAP {r['fap']:.4f}")


if __name__ == "__main__":
    main()
