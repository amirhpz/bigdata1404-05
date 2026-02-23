"""
Feature: reversal_return_5
Class: A (open-domain, signed)

Raw definition:
    -(close/close[-5] - 1)

Normalization policy:
- Robust z-score (past-only) + Gaussian clip + scale to [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from feature_utils import add_feature_per_symbol, class_a_signed


FEATURE_COL = "reversal_return_5_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, lookback: int, norm_window: int) -> pd.Series:
    raw = -(g["close"] / g["close"].shift(lookback) - 1.0)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: reversal_return_5 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.lookback, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()