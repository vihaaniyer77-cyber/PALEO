"""Resumable training-grid downloader (notebook 09 provenance step, repo
version).

Walks the candidate list from select_training_stars.py in order; for each
TIC, fetches one sector of official-pipeline photometry (same ingestion
rules as inference -- see run_pipeline.fetch_lightcurve), resamples to the
10-minute grid, and saves TIC_<id>.npz. Stops once n_train + n_dev grids
exist; the first n_train successes (in candidate order) are the training
set, the next n_dev the dev set (split recorded in split.csv).

Resumable: re-running skips TICs whose npz already exists.

Usage:
    python scripts/download_grids.py --candidates nn_candidates.csv \
        --outdir data/train_grids --n-train 330 --n-dev 30
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paleo.data_utils import resample_uniform          # noqa: E402
from run_pipeline import fetch_lightcurve              # noqa: E402


def main(candidates_csv, outdir, n_train=330, n_dev=30):
    os.makedirs(outdir, exist_ok=True)
    cand = pd.read_csv(candidates_csv)
    need = n_train + n_dev
    done, order = 0, []
    for _, row in cand.iterrows():
        tic = int(row.TICID)
        path = os.path.join(outdir, f"TIC_{tic}.npz")
        if os.path.exists(path):
            order.append(tic)
            done += 1
        else:
            try:
                time, flux, author, sec = fetch_lightcurve(tic)
                g = resample_uniform(time, flux)
                np.savez(path, t0=g["t0"], flux=g["flux"], gap=g["gap"],
                         sigma=g["sigma"], step=g["step"])
                order.append(tic)
                done += 1
                print(f"[{done}/{need}] TIC {tic}: OK ({author} s{sec}, "
                      f"{len(g['flux'])} cells)", flush=True)
            except Exception as e:
                print(f"        TIC {tic}: SKIP ({e})", flush=True)
                continue
        if done >= need:
            break
    if done < need:
        print(f"WARNING: only {done}/{need} grids -- candidate list exhausted")
    split = pd.DataFrame(dict(
        TICID=order,
        split=["train"] * min(n_train, len(order))
              + ["dev"] * max(0, len(order) - n_train)))
    split.to_csv(os.path.join(outdir, "split.csv"), index=False)
    print(f"split written: {min(n_train, len(order))} train / "
          f"{max(0, len(order)-n_train)} dev")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--outdir", default="data/grids")
    ap.add_argument("--n-train", type=int, default=330)
    ap.add_argument("--n-dev", type=int, default=30)
    a = ap.parse_args()
    main(a.candidates, a.outdir, a.n_train, a.n_dev)
