"""
Feature: breakout_strength_20_14
Class: A (positive-domain)

Raw:
    prev20_high = rolling_max(high.shift(1), 20)   # past resistance
    x = max(0, close - prev20_high) / ATR(14)

Process:
    class_a_positive(raw, norm_window)

Output:
    breakout_strength_20_14_scaled_pos_0_1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from features.feature_utils import atr, class_a_positive, add_feature_per_symbol, EPS


FEATURE_COL = "breakout_strength_20_14_scaled_pos_0_1"


def compute_feature(df: pd.DataFrame, norm_window: int = 100) -> pd.Series:
    prev20_high = df["high"].shift(1).rolling(window=20, min_periods=20).max()
    atr14 = atr(df, period=14)
    raw = np.maximum(0.0, df["close"] - prev20_high) / (atr14 + EPS)
    return class_a_positive(raw, norm_window=norm_window)


def build_feature_table(df: pd.DataFrame, norm_window: int = 100) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, norm_window=norm_window))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute breakout_strength_20_14 (Class A) -> [0,1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-window", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}\nColumns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()