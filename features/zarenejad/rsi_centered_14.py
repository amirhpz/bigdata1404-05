"""
Feature: rsi_centered_14
Class: B (bounded semantic oscillator)

Raw definition:
    RSI(14) / 100 - 0.5

Normalization policy:
- No z-score (semantic bounded oscillator)
- No Gaussian clipping
- Keep semantic bounded transform and apply strict clip to [-1, +1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_utils import add_feature_per_symbol, rsi

FEATURE_COL = "rsi_centered_14_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, period: int) -> pd.Series:
    raw_rsi = rsi(g["close"], period=period)  # expected in [0, 100]
    out = raw_rsi / 100.0 - 0.5             # expected in [-1, 1]
    return out.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute rsi_centered_14 (Class B).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=14)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df=df,
        feature_col=FEATURE_COL,
        compute_fn=lambda g: compute_feature(g, args.period),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:")
    print(f" - {FEATURE_COL}")


if __name__ == "__main__":
    main()