"""
Feature: mfi_centered_14
Class: B (bounded semantic oscillator, signed)

Sheet definition:
    MFI14/100 - 0.5

To match final [-1, 1] semantic scale (like RSI reversion feature):
    scaled = 2 * (MFI/100 - 0.5) = (MFI - 50) / 50

MFI formula (standard):
- Typical price TP = (H + L + C)/3
- Raw money flow RMF = TP * Volume
- Positive flow if TP > TP[-1], negative if TP < TP[-1]
- Money Flow Ratio = sum(pos, n) / (sum(neg, n) + eps)
- MFI = 100 - 100/(1 + ratio)

Normalization policy:
- No z-score, no robust z-score
- No Gaussian clipping
- Keep semantic bounded value in [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_utils import EPS, add_feature_per_symbol


FEATURE_COL = "mfi_centered_14_semantic_signed_m1_p1"


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rmf = tp * df["volume"].astype("float64")

    tp_prev = tp.shift(1)
    pos = np.where(tp > tp_prev, rmf, 0.0)
    neg = np.where(tp < tp_prev, rmf, 0.0)

    pos_s = pd.Series(pos, index=df.index).rolling(window=period, min_periods=period).sum()
    neg_s = pd.Series(neg, index=df.index).rolling(window=period, min_periods=period).sum()

    ratio = pos_s / (neg_s + EPS)
    mfi_ = 100.0 - (100.0 / (1.0 + ratio))
    return mfi_.clip(0.0, 100.0)


def compute_feature(g: pd.DataFrame, period: int) -> pd.Series:
    m = mfi(g, period=period)  # [0, 100]
    raw = (m / 100.0) - 0.5    # [-0.5, 0.5]
    scaled = 2.0 * raw         # [-1, 1]
    return scaled.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: mfi_centered_14 (Class B signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=14)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, args.period))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()