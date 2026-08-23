"""Train the PALEO Stage-1 U-Net (as run in notebook 10, 2026-08-08).

Data: resampled 10-min grids produced by scripts/download_training_set.py
(notebook 09). Training examples are generated on the fly - every step injects
fresh randomized trapezoidal transits, so no example ever repeats; stars are
the finite resource (learning curve saturated at ~200 stars).

Usage: python train.py --base /path/to/PALEO_colab [--steps 8000]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paleo.data_utils import make_training_sample
from paleo.model import UNet1D, masked_bce


def load_grids(base, folder):
    out = []
    for fn in sorted(os.listdir(f"{base}/data/{folder}")):
        z = np.load(f"{base}/data/{folder}/{fn}")
        out.append(dict(flux=z["flux"], gap=z["gap"], sigma=float(z["sigma"]),
                        t0=float(z["t0"]), step=float(z["step"]), tic=fn))
    return out


def batch(grids, rng, device, bs=16):
    samples = [make_training_sample(grids[rng.integers(len(grids))], rng)
               for _ in range(bs)]
    n_max = max(s[0].shape[1] for s in samples)
    n_max += (-n_max) % 8
    X = np.zeros((bs, 2, n_max), np.float32)
    Y = np.zeros((bs, n_max), np.float32)
    G = np.ones((bs, n_max), np.float32)          # padding counts as gap
    for i, (x, y, _) in enumerate(samples):
        X[i, :, :x.shape[1]] = x
        Y[i, :y.shape[0]] = y
        G[i, :x.shape[1]] = x[1]
    return (torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device),
            torch.from_numpy(G).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_grids = load_grids(args.base, "train_grids")
    dev_grids = load_grids(args.base, "dev_grids")
    print(f"{len(train_grids)} train, {len(dev_grids)} dev on {device}")

    dev_rng = np.random.default_rng(999)
    DEV_SET = [batch(dev_grids, dev_rng, device) for _ in range(12)]

    def dev_loss(model):
        model.eval()
        with torch.no_grad():
            losses = [float(masked_bce(model(X), Y, G)) for X, Y, G in DEV_SET]
        model.train()
        return float(np.mean(losses))

    ckpt = f"{args.base}/models/paleo_unet_v1.pt"
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = UNet1D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best, stale, hist = 0, np.inf, 0, []

    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best, hist = ck["step"], ck["best"], ck["hist"]
        print(f"resumed at step {start}, best dev {best:.4f}")

    for step in range(start, args.steps):
        X, Y, G = batch(train_grids, rng, device)
        loss = masked_bce(model(X), Y, G)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            dl = dev_loss(model)
            hist.append(dict(step=step + 1, train=float(loss.detach()), dev=dl))
            if dl < best - 1e-4:
                best, stale = dl, 0
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                                step=step + 1, best=best, hist=hist), ckpt)
                flag = "* saved"
            else:
                stale += 1
                flag = f"(stale {stale}/{args.patience})"
            print(f"step {step+1}/{args.steps} train={float(loss.detach()):.3f} "
                  f"dev={dl:.4f} {flag}")
            if stale >= args.patience:
                print("early stop")
                break
    pd.DataFrame(hist).to_csv(f"{args.base}/results/nn_training_history.csv", index=False)
    print(f"best dev loss {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
