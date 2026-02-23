"""
Feature: rsi_reversion_14
Class: B (bounded semantic oscillator, signed)

Sheet definition:
    0.5 - RSI14/100

To match final scale [-1, 1] (semantic):
    scaled = 2 * (0.5 - RSI14/100) = (50 - RSI14) / 50

Normalization policy:
- No z-score, no Gaussian clipping
- Keep bounded semantic value in [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from feature_utils import add_feature_per_symbol, rsi


FEATURE_COL = "rsi_reversion_14_semantic_signed_m1_p1"


def compute_feature(g: pd.DataFrame, period: int) -> pd.Series:
    r = rsi(g["close"], period=period)  # [0, 100]
    raw = (50.0 - r) / 50.0
    return raw.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: rsi_reversion_14 (Class B signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=14)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, args.period))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()