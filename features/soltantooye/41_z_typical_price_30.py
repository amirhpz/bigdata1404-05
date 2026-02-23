"""
Feature: z_typical_price_30
Class: A
Raw: (TP-mean(TP,30))/(std(TP,30)+eps) - standardized deviation
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "z_typical_price_30_scaled_signed_m1_p1"
GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416


def compute_feature(g: pd.DataFrame, period: int) -> pd.Series:
    # Current typical price
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    
    # Past-only statistics
    past_tp = tp.shift(1)
    mean = past_tp.rolling(window=period, min_periods=period).mean()
    std = past_tp.rolling(window=period, min_periods=period).std()
    
    # Raw standardized deviation (already a z-score)
    raw_z = (tp - mean) / (std + EPS)
    
    # Only clip and scale (no robust z!)
    clipped = raw_z.clip(GAUSSIAN_CLIP_LOW, GAUSSIAN_CLIP_HIGH)
    scaled = clipped / GAUSSIAN_CLIP_HIGH
    
    return scaled


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: z_typical_price_30")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=30)
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