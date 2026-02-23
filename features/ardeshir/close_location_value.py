"""
Feature: close_location_value
Class: B (bounded semantic position)

Raw/semantic:
    (close - low) / (high - low + eps) -> [0,1]

Rules:
- NO z-score
- NO robust z-score
- NO Gaussian clipping

Output:
    close_location_value_pos_0_1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from features.feature_utils import add_feature_per_symbol, EPS


FEATURE_COL = "close_location_value_pos_0_1"


def compute_feature(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"])
    x = (df["close"] - df["low"]) / (rng + EPS)
    return x.clip(0.0, 1.0)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, compute_feature)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute close_location_value (Class B) -> [0,1].")
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