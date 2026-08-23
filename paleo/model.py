"""PALEO Stage-1 neural detector: small 1-D U-Net for per-timestep transit
segmentation. Written by notebook 10, imported by 10-12."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """~1.9M params. Input (B, 2, N) [flux/sigma, gap]; output (B, N) logits.
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


def masked_bce(logits, target, gap, pos_weight=25.0):
    """BCE over non-gap cells only, up-weighting the rare in-transit class."""
    w = (gap < 0.5).float()
    pw = torch.full_like(target, 1.0) + (pos_weight - 1.0) * target
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=pw * w,
                                              reduction="sum")
    return loss / w.sum().clamp(min=1.0)
