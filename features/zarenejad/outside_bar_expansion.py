"""
Feature: outside_bar_expansion
Class: A (open-domain, positive)

Raw definition:
    I[high >= high[-1] and low <= low[-1]] * (high - low) / (ATR(14) + eps)

Normalization policy:
- Robust z-score (past-only, via utility)
- Gaussian central-60% clipping to [-0.8416, +0.8416]
- Fixed scaling to positive range [0, 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_utils import EPS, add_feature_per_symbol, atr, class_a_positive

FEATURE_COL = "outside_bar_expansion_scaled_pos_0_1"


def compute_feature(g: pd.DataFrame, atr_period: int, norm_window: int) -> pd.Series:
    prev_high = g["high"].shift(1)
    prev_low = g["low"].shift(1)

    outside = ((g["high"] >= prev_high) & (g["low"] <= prev_low)).astype(float)
    raw = outside * (g["high"] - g["low"]) / (atr(g, period=atr_period) + EPS)

    return class_a_positive(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute outside_bar_expansion (Class A positive).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df=df,
        feature_col=FEATURE_COL,
        compute_fn=lambda g: compute_feature(g, args.atr_period, args.norm_window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:")
    print(f" - {FEATURE_COL}")


if __name__ == "__main__":
    main()