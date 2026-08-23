"""Test 2: control run -- the pipeline must stay quiet on the same star with
no injection.

Runs the identical chain on the untouched light curve of TIC 272566345 and
asserts no detection at FAP < 0.05. (The star has no known transiting
planet; its intrinsic bumps must not be promoted to a claim.)

Requires local data + checkpoint (not in the repo):
    PALEO_GRID  = path to TIC_272566345.npz        (dev grid)
    PALEO_MODEL = path to paleo_unet_v1.pt

Run:  python tests/test_control_quiet.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paleo.detect import (DUR_DEFAULT, knee_cut_scores, load_model,  # noqa: E402
                          nn_events, nn_score)
from paleo.stage2 import stage2_search                            # noqa: E402

GRID = os.environ.get("PALEO_GRID", "data/dev_grids/TIC_272566345.npz")
MODEL = os.environ.get("PALEO_MODEL", "models/paleo_unet_v1.pt")


def main():
    d = np.load(GRID)
    grid = dict(t0=float(d["t0"]), flux=d["flux"].astype(float),
                gap=d["gap"].astype(float), sigma=float(d["sigma"]),
                step=float(d["step"]))
    t = grid["t0"] + np.arange(len(grid["flux"])) * grid["step"]

    model = load_model(MODEL, "cpu")
    score = nn_score(model, grid["flux"], grid["gap"], grid["sigma"])
    times, scores = nn_events(score, grid)
    cut = knee_cut_scores(scores) if times else 0
    r = stage2_search(times[:cut], t[0], t[-1], tol=DUR_DEFAULT,
                      rng=np.random.default_rng(0))

    fired = r["period"] is not None and r["fap"] < 0.05
    assert not fired, (f"FALSE ALARM: control fired at P={r['period']} "
                       f"with FAP={r['fap']:.4f}")
    fp = f"{r['period']:.3f} d" if r["period"] else "none"
    print(f"PASS: control stays quiet | best alignment {fp}, "
          f"FAP {r['fap']:.4f} (>= 0.05) | max score {max(scores):.3f}")


if __name__ == "__main__":
    main()
