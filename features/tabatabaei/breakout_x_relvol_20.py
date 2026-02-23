"""
Feature: breakout_x_relvol_20
Class: A (open-domain, signed)

Raw definition:
    ((close - prev20High) / (ATR(14) + eps)) * (volume / (EMA(volume, 20) + eps))

Where:
    prev20High_t = max(high_{t-20}..high_{t-1})   (past-only)

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


FEATURE_COL = "breakout_x_relvol_20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, high_window: int, atr_period: int, vol_ema_span: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    # Past-only rolling high to avoid leakage (t uses highs from t-window .. t-1).
    prev_high = g["high"].shift(1).rolling(window=high_window, min_periods=high_window).max()

    # Causal ATR.
    atr_n = atr(g, atr_period)

    # Causal volume EMA.
    vol_ema = g["volume"].ewm(span=vol_ema_span, adjust=False, min_periods=vol_ema_span).mean()
    rel_vol = g["volume"] / (vol_ema + EPS)

    raw = ((g["close"] - prev_high) / (atr_n + EPS)) * rel_vol
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: breakout_x_relvol_20 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--high-window", type=int, default=20)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--vol-ema-span", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.high_window, args.atr_period, args.vol_ema_span, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()