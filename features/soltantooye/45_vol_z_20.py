"""
Feature: vol_z_20
Class: A
Raw: (V-mean(V,20))/(std(V,20)+eps) - rolling z-score
Then only clip and scale (no robust z!)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "vol_z_20_scaled_signed_m1_p1"
GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416


def compute_feature(g: pd.DataFrame, period: int) -> pd.Series:
    # Past-only volume statistics
    past_vol = g["volume"].shift(1)
    mean = past_vol.rolling(window=period, min_periods=period).mean()
    std = past_vol.rolling(window=period, min_periods=period).std()
    
    # Raw z-score (already normalized)
    raw_z = (g["volume"] - mean) / (std + EPS)
    
    # Only clip and scale (no robust z!)
    clipped = raw_z.clip(GAUSSIAN_CLIP_LOW, GAUSSIAN_CLIP_HIGH)
    scaled = clipped / GAUSSIAN_CLIP_HIGH
    
    return scaled


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: vol_z_20")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df, FEATURE_COL,
        lambda g: compute_feature(g, args.period),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Column: {FEATURE_COL}")


if __name__ == "__main__":
    main()