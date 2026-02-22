"""
Feature: rsi_reversion_14
Class: B (bounded semantic oscillator)
Raw definition:
    Distance from RSI center: (0.5-RSI/100)
Normalization policy:
- Semantic RSI relative transform to [-1, +1]
- No z-score, no Gaussian clipping
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import add_feature_per_symbol, rsi

FEATURE_COL = "rsi_reversion_14_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, rsi_period: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    rsi_val = rsi(g["close"], rsi_period)
    # Semantic transform: (RSI - 50) / 50 to get [-1, +1]
    return (rsi_val - 50.0) / 50.0


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: rsi_reversion_14 (Class B)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rsi-period", type=int, default=14)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.rsi_period),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()