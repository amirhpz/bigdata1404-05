"""
Feature: dist_prev20_high_atr
Class: A (open-domain, signed)
Raw definition:
    Distance to prior 20-bar high: (C-roll_max(H,20)[t-1])/ATR_14
Normalization policy:
- Robust z-score (past-only, via utility)
- Gaussian central-60% clipping to [-0.8416, +0.8416]
- Fixed scaling to signed range [-1, +1]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, atr, class_a_signed

FEATURE_COL = "dist_prev20_high_atr_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, lookback: int, atr_period: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    past_high = g["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
    dist = g["close"] - past_high
    raw = dist / (atr(g, atr_period) + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: dist_prev20_high_atr (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.lookback, args.atr_period, args.norm_window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()