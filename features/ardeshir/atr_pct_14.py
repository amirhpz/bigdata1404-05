"""
Feature: atr_pct_14
Class: A (positive-domain)

Raw:
    ATR(14) / close

Process:
    class_a_positive(raw, norm_window)

Output:
    atr_pct_14_scaled_pos_0_1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from features.feature_utils import atr, class_a_positive, add_feature_per_symbol, EPS


FEATURE_COL = "atr_pct_14_scaled_pos_0_1"


def compute_feature(df: pd.DataFrame, norm_window: int = 100) -> pd.Series:
    raw = atr(df, period=14) / (df["close"] + EPS)
    return class_a_positive(raw, norm_window=norm_window)


def build_feature_table(df: pd.DataFrame, norm_window: int = 100) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, norm_window=norm_window))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute atr_pct_14 (Class A) -> [0,1].")
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