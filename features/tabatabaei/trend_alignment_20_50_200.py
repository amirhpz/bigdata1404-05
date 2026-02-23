"""
Feature: trend_alignment_20_50_200
Class: A (open-domain, signed)

Raw definition:
    ((EMA20 - EMA50) / (ATR14 + eps)) * ((EMA50 - EMA200) / (ATR14 + eps))

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


FEATURE_COL = "trend_alignment_20_50_200_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, ema20: int, ema50: int, ema200: int, atr_period: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    e20 = g["close"].ewm(span=ema20, adjust=False, min_periods=ema20).mean()
    e50 = g["close"].ewm(span=ema50, adjust=False, min_periods=ema50).mean()
    e200 = g["close"].ewm(span=ema200, adjust=False, min_periods=ema200).mean()

    atr_n = atr(g, atr_period)

    gap_short = (e20 - e50) / (atr_n + EPS)
    gap_long = (e50 - e200) / (atr_n + EPS)

    raw = gap_short * gap_long
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: trend_alignment_20_50_200 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ema20", type=int, default=20)
    p.add_argument("--ema50", type=int, default=50)
    p.add_argument("--ema200", type=int, default=200)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.ema20, args.ema50, args.ema200, args.atr_period, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()