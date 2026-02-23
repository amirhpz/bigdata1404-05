"""
Feature: rsi_reversion_14
Class: B (bounded semantic oscillator)
Raw: Distance from RSI center
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import add_feature_per_symbol, rsi

FEATURE_COL = "rsi_reversion_14_semantic_signed_m1_p1"


def compute_feature(g: pd.DataFrame, rsi_period: int) -> pd.Series:
    rsi_val = rsi(g["close"], rsi_period)
    # Semantic transform: (RSI - 50) / 50 to [-1, +1]
    return (rsi_val - 50.0) / 50.0


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: rsi_reversion_14")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rsi-period", type=int, default=14)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df, FEATURE_COL,
        lambda g: compute_feature(g, args.rsi_period),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Column: {FEATURE_COL}")


if __name__ == "__main__":
    main()