"""
Feature: upper_wick_to_tr
Class: B (bounded semantic ratio)

Raw/semantic:
    (high - max(open, close)) / (TR + eps) -> [0,1]

Rules:
- NO z-score
- NO robust z-score
- NO Gaussian clipping

Output:
    upper_wick_to_tr_pos_0_1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from features.feature_utils import true_range, add_feature_per_symbol, EPS


FEATURE_COL = "upper_wick_to_tr_pos_0_1"


def compute_feature(df: pd.DataFrame) -> pd.Series:
    tr = true_range(df)
    upper = df["high"] - np.maximum(df["open"], df["close"])
    x = upper / (tr + EPS)
    return x.clip(0.0, 1.0)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, compute_feature)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute upper_wick_to_tr (Class B) -> [0,1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}\nColumns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()