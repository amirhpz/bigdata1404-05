"""
Feature script: percent_b_20

Class: B (Bounded semantic oscillator / Bollinger Band position)
Final scale: [-1, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Measures where the current close sits within the 20-bar Bollinger Band.
- %B = 0 means price is at lower band, %B = 1 means price is at upper band.
- Transformed to [-1, 1] via the standard Class B semantic: 2*(%B - 0.5).
- Useful for mean-reversion signals: -1 = deeply oversold, +1 = deeply overbought.

Raw formula (causal, past-only):
    SMA_20_t   = mean(close_{t-20}..close_{t-1})     (PAST-ONLY via shift(1))
    std_20_t   = std(close_{t-20}..close_{t-1})
    BB_up_20_t = SMA_20_t + 2 * std_20_t
    BB_dn_20_t = SMA_20_t - 2 * std_20_t

    pct_b_raw_t = (close_t - BB_dn_20_t) / (BB_up_20_t - BB_dn_20_t + eps)  ∈ [0, 1]

    percent_b_20_t = 2 * (pct_b_raw_t - 0.5)   ∈ [-1, 1]

Notes:
- No z-score or Gaussian clipping (Class B).
- Mild semantic clip to [-1, 1] enforced to handle rare extreme excursions beyond bands.
- Rolling stats are shift(1)-based (past-only) to avoid leakage.
- Computation is per-symbol; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "percent_b_20"
BB_MULTIPLIER = 2.0


def compute_feature(g: pd.DataFrame, window: int = 20) -> pd.Series:
    """Bollinger Band %B transformed to [-1, 1]."""
    past_close = g["close"].shift(1)
    sma = past_close.rolling(window=window, min_periods=window).mean()
    std = past_close.rolling(window=window, min_periods=window).std(ddof=1)
    bb_up = sma + BB_MULTIPLIER * std
    bb_dn = sma - BB_MULTIPLIER * std
    pct_b_raw = (g["close"] - bb_dn) / ((bb_up - bb_dn) + EPS)
    raw_signed = 2.0 * (pct_b_raw - 0.5)
    return raw_signed.clip(lower=-1.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, window))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute percent_b_20 (Class B) — Bollinger Band %B transformed to [-1, 1]."
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
