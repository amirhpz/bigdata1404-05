"""
Feature: macd_hist_atr
Class: A (open-domain, signed)

Raw definition:
    (MACD(12,26) - Signal(9)) / (ATR(14) + eps)

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

FEATURE_COL = "macd_hist_atr_scaled_signed_m1_p1"


def ema(series: pd.Series, span: int) -> pd.Series:
    # Causal EMA with warm-up NaNs
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_feature(g: pd.DataFrame, norm_window: int) -> pd.Series:
    e12 = ema(g["close"], span=12)
    e26 = ema(g["close"], span=26)
    macd_line = e12 - e26

    signal9 = ema(macd_line, span=9)
    hist = macd_line - signal9  # MACD histogram

    a14 = atr(g, period=14)
    raw = hist / (a14 + EPS)

    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute macd_hist_atr (Class A signed).")
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