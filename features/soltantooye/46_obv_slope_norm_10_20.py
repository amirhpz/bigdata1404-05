"""
Feature: obv_slope_norm_10_20
Class: B (bounded semantic)
Raw definition:
    OBV directional pressure: (OBV-OBV[t-10])/(std(V,20)+eps)
Normalization policy:
- Semantic bounded signed flow ratio
- No z-score, no Gaussian clipping
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "obv_slope_norm_10_20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, slope_period: int, vol_period: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    # Compute OBV
    sign = (g["close"] - g["close"].shift(1)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (sign * g["volume"]).cumsum()
    
    # OBV slope
    obv_slope = obv - obv.shift(slope_period)
    
    # Normalize by volume std
    past_vol = g["volume"].shift(1)
    vol_std = past_vol.rolling(window=vol_period, min_periods=vol_period).std()
    
    return obv_slope / (vol_std + EPS)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: obv_slope_norm_10_20 (Class B)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--slope-period", type=int, default=10)
    p.add_argument("--vol-period", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.slope_period, args.vol_period),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()