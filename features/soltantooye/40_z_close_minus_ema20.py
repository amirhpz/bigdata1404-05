"""
Feature: z_close_minus_ema20
Class: A (open-domain, signed)  [Robust*]

Sheet intent:
    (C - EMA20) / (std(C - EMA20, 20) + eps)

Implementation per protocol:
- We use Class A robust z-score on the raw deviation (C - EMA20),
  which replaces std-based z with leakage-safe robust z (median/MAD).

Raw definition used:
    dev = close - EMA(close, 20)

Normalization policy:
- Robust z-score (past-only) with norm_window default=20 (matches sheet's 20-ish standardization)
- Gaussian clip + fixed scaling to [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from feature_utils import add_feature_per_symbol, class_a_signed


FEATURE_COL = "z_close_minus_ema20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, ema_period: int, norm_window: int) -> pd.Series:
    ema = g["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    dev = g["close"] - ema
    return class_a_signed(dev, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: z_close_minus_ema20 (Class A signed, robust*)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ema-period", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=20)  # aligns with sheet's 20-style z
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.ema_period, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()