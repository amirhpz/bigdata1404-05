"""
Feature: up_close_ratio_5
Class: B (bounded semantic frequency)

Raw definition:
    count(close_i > close_{i-1}, window=5) / 5

Normalization policy:
- No z-score (semantic bounded frequency)
- No Gaussian clipping
- Keep bounded ratio and apply strict clip to [0, 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_utils import add_feature_per_symbol

FEATURE_COL = "up_close_ratio_5_scaled_pos_0_1"


def compute_feature(g: pd.DataFrame, window: int) -> pd.Series:
    up = (g["close"] > g["close"].shift(1)).astype(float)
    raw = up.rolling(window=window, min_periods=window).mean()
    return raw.clip(0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute up_close_ratio_5 (Class B bounded).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df=df,
        feature_col=FEATURE_COL,
        compute_fn=lambda g: compute_feature(g, args.window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:")
    print(f" - {FEATURE_COL}")


if __name__ == "__main__":
    main()