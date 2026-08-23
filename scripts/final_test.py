"""The one-shot final test (notebook 12, PASSED 2026-08-08).

Runs the frozen NN pipeline on the quarantined List A/B stars with the
seed-20260807 pre-registered injection plan - the identical injections the GP
prototype and CETRA faced - plus paired no-injection controls, applies the
paired-control exclusion rule, and scores against the pre-registered criteria.

Pre-registered criteria (fixed before the NN was trained; control criterion
amended before this test ran, on dev evidence of a calibrated FAP):
  * strict recall >= 10/20 (FAP < 0.05, period error < 5%)
  * controls, post paired-control exclusion: <= 1 at FAP < 0.05, 0 at FAP < 0.01

Result: 11/20 strict, 0/16 controls at 0.05, 0/16 at 0.01 -> PASSED.

Usage: python final_test.py --base /path/to/PALEO_colab
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paleo.data_utils import resample_uniform, inject_periodic
from paleo.detect import load_model, nn_score, nn_events, knee_cut_scores, DUR_DEFAULT
from paleo.stage2 import stage2_search

DUR = DUR_DEFAULT
PLAN_SEED = 20260807


def build_plan(base):
    """Regenerate the pre-registered plan; MUST match fresh_validation.csv."""
    list_a = pd.read_csv(f"{base}/results/target_lists/list_a_rotation_validation.csv")
    list_b = pd.read_csv(f"{base}/results/target_lists/list_b_activity_stratified.csv")
    rng = np.random.default_rng(PLAN_SEED)
    targets = [(int(t), f"{base}/data/list_a_lightcurves")
               for t in list_a.loc[list_a.published_period > 1, "TICID"]] + \
              [(int(t), f"{base}/data/list_b_lightcurves")
               for t in list_b.loc[list_b.adopted_period > 1, "TICID"]]
    plan = [(tic, d, round(float(rng.uniform(3.0, 9.0)), 3),
             round(float(rng.uniform(0.1, 0.9)), 3)) for tic, d in targets]
    ref = pd.read_csv(f"{base}/results/fresh_validation.csv")
    ref = ref[ref.true_P.notna()].set_index("tic").true_P.to_dict()
    for tic, _, P, _ in plan:
        if tic in ref:
            assert abs(ref[tic] - P) < 1e-6, f"PLAN MISMATCH TIC {tic}"
    return plan


def nn_trial(model, device, tic, lc_dir, true_P, ef):
    d = np.load(f"{lc_dir}/TIC_{int(tic)}.npz")
    g = resample_uniform(d["time"], d["flux"])
    if true_P is not None:
        flux, mask, n_true = inject_periodic(g, true_P, ef, 0.01, DUR, 0.0, shape="box")
    else:
        flux, n_true = g["flux"].copy(), 0
    score = nn_score(model, flux, g["gap"], g["sigma"], device)
    times, scores = nn_events(score, g)
    ev = times[:knee_cut_scores(scores)] if times else []
    t = g["t0"] + np.arange(len(g["flux"])) * g["step"]
    r2 = stage2_search(ev, t[0], t[-1], tol=DUR, rng=np.random.default_rng(0))
    return dict(tic=int(tic), true_P=true_P, n_true=n_true, n_events=len(ev),
                found_P=r2["period"], n_matched=r2["n_matched"], fap=r2["fap"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(f"{args.base}/models/paleo_unet_v1.pt", device)
    plan = build_plan(args.base)
    print(f"plan verified: {len(plan)} targets")

    out_csv = f"{args.base}/results/final_test_nn.csv"
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv).apply(
            lambda r: (int(r.tic), r.true_P if pd.notna(r.true_P) else -1), axis=1))
    for i, (tic, lcd, P, ef) in enumerate(plan):
        for true_P in (P, None):
            if (tic, true_P if true_P else -1) in done:
                continue
            r = nn_trial(model, device, tic, lcd, true_P, ef)
            pd.DataFrame([r]).to_csv(out_csv, mode="a",
                                     header=not os.path.exists(out_csv), index=False)
            fp = f"{r['found_P']:.3f}" if r["found_P"] else "None"
            print(f"[{i+1}/{len(plan)}] TIC {tic} true_P={true_P}: "
                  f"found={fp}, FAP={r['fap']:.3f}")

    df = pd.read_csv(out_csv)
    inj = df[df.true_P.notna()].copy()
    inj["err"] = abs(inj.found_P - inj.true_P) / inj.true_P * 100
    ctrl = df[df.true_P.isna()].copy()
    merged = inj.merge(ctrl[["tic", "found_P", "fap"]], on="tic",
                       suffixes=("", "_ctrl"))
    intrinsic = merged[merged.found_P_ctrl.notna() & merged.found_P.notna() &
                       (abs(merged.found_P - merged.found_P_ctrl)
                        / merged.found_P < 0.05)].tic.tolist()
    clean_ctrl = ctrl[~ctrl.tic.isin(intrinsic)]
    win = inj[~inj.tic.isin(intrinsic) & (inj.n_true >= 3)]
    print(f"\nintrinsic-signal exclusions (paired control): {intrinsic}")
    print(f"CONFIRMED: {((inj.fap < .05) & (inj.err.fillna(99) < 5)).sum()}/{len(inj)} raw | "
          f"{((win.fap < .05) & (win.err.fillna(99) < 5)).sum()}/{len(win)} clean winnable")
    print(f"controls: {(clean_ctrl.fap < .05).sum()}/{len(clean_ctrl)} at 0.05, "
          f"{(clean_ctrl.fap < .01).sum()}/{len(clean_ctrl)} at 0.01")
    print("bar: >=10/20 strict; <=1 control at 0.05, 0 at 0.01")


if __name__ == "__main__":
    main()
