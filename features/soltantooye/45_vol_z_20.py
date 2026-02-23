"""
Feature: vol_z_20
Class: A (open-domain, signed) [Robust*]

Sheet intent:
    (V - mean(V, 20)) / (std(V, 20) + eps)

Implementation per protocol:
- Define a relative 'surprise' raw feature (unitless) using past-only mean:
    raw = V/(mean_past_20 + eps) - 1
  where mean_past_20 is computed from V.shift(1) to avoid self-inclusion.
- Then apply Class A robust z-score (past-only) + Gaussian clip + scale.

Normalization policy:
- Robust z-score (past-only) + Gaussian clip + scale to [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from feature_utils import EPS, add_feature_per_symbol, class_a_signed


FEATURE_COL = "vol_z_20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, lookback: int, norm_window: int) -> pd.Series:
    mean_past = g["volume"].shift(1).rolling(window=lookback, min_periods=lookback).mean()
    raw = g["volume"] / (mean_past + EPS) - 1.0
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: vol_z_20 (Class A signed, robust*)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=20)
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