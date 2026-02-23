"""
Feature: ema_gap_atr_20
Class: A (open-domain, signed)

Raw definition:
    (close - EMA(close, 20)) / ATR(14)

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

FEATURE_COL = "ema_gap_atr_20_scaled_signed_m1_p1"


def ema(series: pd.Series, span: int) -> pd.Series:
    # Causal EMA; keep warm-up NaNs.
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_feature(g: pd.DataFrame, norm_window: int) -> pd.Series:
    e20 = ema(g["close"], span=20)
    a14 = atr(g, period=14)
    raw = (g["close"] - e20) / (a14 + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute ema_gap_atr_20 (Class A signed).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df=df,
        feature_col=FEATURE_COL,
        compute_fn=lambda g: compute_feature(g, args.norm_window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:")
    print(f" - {FEATURE_COL}")


if __name__ == "__main__":
    main()