"""
Feature: mom_vol_norm_20_60
Class: A (open-domain, signed)

Raw definition:
    (close / close[-20] - 1) / (std(ret1, 60) + eps)

Where (vol is past-only):
    ret1_t = close_t / close_{t-1} - 1
    std(ret1,60)_t = std(ret1_{t-60}..ret1_{t-1})

Normalization policy:
- Robust z-score (past-only, via utility)
- Gaussian central-60% clipping to [-0.8416, +0.8416]
- Fixed scaling to signed range [-1, +1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, class_a_signed


FEATURE_COL = "mom_vol_norm_20_60_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, mom_lookback: int, vol_lookback: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    mom = (g["close"] / (g["close"].shift(mom_lookback) + EPS)) - 1.0

    prev_close = g["close"].shift(1)
    ret1 = (g["close"] / (prev_close + EPS)) - 1.0

    # Past-only rolling std to avoid leakage.
    vol = ret1.shift(1).rolling(window=vol_lookback, min_periods=vol_lookback).std()

    raw = mom / (vol + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: mom_vol_norm_20_60 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mom-lookback", type=int, default=20)
    p.add_argument("--vol-lookback", type=int, default=60)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.mom_lookback, args.vol_lookback, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()