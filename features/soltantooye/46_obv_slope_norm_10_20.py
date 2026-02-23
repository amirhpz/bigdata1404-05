"""
Feature: obv_slope_norm_10_20
Class: B (bounded semantic flow ratio, signed)

Raw definition:
    OBV = cumulative sum( sign(close - close[-1]) * volume )
    (OBV - OBV[-10]) / (sum(volume, 20) + eps)

Leakage prevention:
- sum(volume, 20) computed on volume.shift(1) so candle t is excluded from denominator window.
  (This matches README rolling-stats past-only rule.)

Normalization policy:
- No z-score, no Gaussian clipping
- Keep semantic bounded ratio in [-1, 1] (clip for numeric safety)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_utils import EPS, add_feature_per_symbol


FEATURE_COL = "obv_slope_norm_10_20_semantic_signed_m1_p1"


def compute_feature(g: pd.DataFrame, obv_lookback: int, vol_norm_lookback: int) -> pd.Series:
    # OBV (per symbol, causal)
    dir_ = np.sign(g["close"].diff()).astype("float64")  # -1,0,1; first row NaN
    dir_ = pd.Series(dir_, index=g.index).fillna(0.0)
    obv = (dir_ * g["volume"].astype("float64")).cumsum()

    num = obv - obv.shift(obv_lookback)

    # Past-only volume sum (exclude current candle)
    denom = g["volume"].shift(1).rolling(window=vol_norm_lookback, min_periods=vol_norm_lookback).sum()

    raw = num / (denom + EPS)
    return raw.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: obv_slope_norm_10_20 (Class B signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--obv-lookback", type=int, default=10)
    p.add_argument("--vol-norm-lookback", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.obv_lookback, args.vol_norm_lookback),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()