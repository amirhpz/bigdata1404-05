"""
Feature: dist_prev20_high_atr
Class: A (open-domain, signed)

Raw definition:
    (close - roll_max(high, 20)[-1]) / ATR(14)

Leakage prevention:
- prior_20_high uses high.shift(1).rolling(20).max() so candle t is excluded.

Normalization policy:
- Robust z-score (past-only) + Gaussian clip + scale to [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from feature_utils import EPS, add_feature_per_symbol, atr, class_a_signed


FEATURE_COL = "dist_prev20_high_atr_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, sr_lookback: int, atr_period: int, norm_window: int) -> pd.Series:
    prior_high = g["high"].shift(1).rolling(window=sr_lookback, min_periods=sr_lookback).max()
    raw = (g["close"] - prior_high) / (atr(g, atr_period) + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: dist_prev20_high_atr (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sr-lookback", type=int, default=20)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.sr_lookback, args.atr_period, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()