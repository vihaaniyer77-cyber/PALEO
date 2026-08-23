"""Training-star selection (notebook 09 provenance step, repo version).

Draws the candidate pool for training/dev stars from the TARS rotation
catalog with the evaluation stars quarantined by TIC ID *before* sampling,
under a fixed seed. Writes nn_candidates.csv for download_grids.py.

Usage:
    python scripts/select_training_stars.py \
        --tars tars_table_2.csv \
        --quarantine list_a.csv list_b.csv \
        --out nn_candidates.csv

TARS catalog (Boyle et al. 2026): zenodo.org/records/19917941
(tars_table_2.csv.zip). Quarantine CSVs need a TICID column.
Historical run: seed 20260808, 700 candidates, pool 510,861 after cuts.
"""
import argparse

import pandas as pd

USECOLS = ["TICID", "teff", "Tmag", "adopted_period", "adopted_period_unc",
           "flag_multiple_periods", "flag_possible_binary", "non_single_star"]


def main(tars_csv, quarantine_csvs, out_csv, seed=20260808, n=700,
         min_prot=1.0):
    tars = pd.read_csv(tars_csv, usecols=USECOLS)
    # single, unambiguous rotation period; no binarity evidence
    clean = tars[(tars.flag_multiple_periods == False)          # noqa: E712
                 & (tars.flag_possible_binary == False)         # noqa: E712
                 & (tars.non_single_star == 0)].dropna(subset=["adopted_period"])

    quarantine = set()
    for q in quarantine_csvs:
        quarantine |= set(pd.read_csv(q).TICID.astype(int))

    pool = clean[(clean.adopted_period > min_prot)
                 & (~clean.TICID.isin(quarantine))]
    print(f"catalog {len(tars)} -> clean {len(clean)} -> "
          f"pool after P_rot>{min_prot} d cut + quarantine: {len(pool)}")

    cand = pool.sample(n=n, random_state=seed)[["TICID", "adopted_period", "Tmag"]]
    assert not set(cand.TICID.astype(int)) & quarantine, "QUARANTINE BREACH"
    cand.to_csv(out_csv, index=False)
    print(f"{n} candidates -> {out_csv} (seed {seed}); download until "
          f"330 train + 30 dev succeed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", required=True)
    ap.add_argument("--quarantine", nargs="+", required=True,
                    help="CSV(s) of evaluation stars to exclude (TICID column)")
    ap.add_argument("--out", default="nn_candidates.csv")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--n", type=int, default=700)
    a = ap.parse_args()
    main(a.tars, a.quarantine, a.out, a.seed, a.n)
