"""
Feature: z_close_minus_ema20
Class: A
Raw: (C-EMA(20))/(std(C,20)+eps) - already a z-score, just need clip+scale
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "z_close_minus_ema20_scaled_signed_m1_p1"
GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416


def compute_feature(g: pd.DataFrame, ema_period: int) -> pd.Series:
    # Past-only EMA
    past_close = g["close"].shift(1)
    ema = past_close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    
    # Past-only rolling std
    std = past_close.rolling(window=ema_period, min_periods=ema_period).std()
    
    # Raw z-score (already normalized)
    raw_z = (g["close"] - ema) / (std + EPS)
    
    # Only clip and scale (no robust z!)
    clipped = raw_z.clip(GAUSSIAN_CLIP_LOW, GAUSSIAN_CLIP_HIGH)
    scaled = clipped / GAUSSIAN_CLIP_HIGH  # to [-1, +1]
    
    return scaled


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: z_close_minus_ema20")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ema-period", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df, FEATURE_COL,
        lambda g: compute_feature(g, args.ema_period),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Column: {FEATURE_COL}")


if __name__ == "__main__":
    main()