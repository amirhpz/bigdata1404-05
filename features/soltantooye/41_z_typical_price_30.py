"""
Feature: z_typical_price_30
Class: A (open-domain, signed)
Raw definition:
    Typical price z: (TP-mean(TP,30))/(std(TP,30)+eps)
    where TP = (H+L+C)/3
Normalization policy:
- Robust z-score (past-only, via utility)
- Gaussian central-60% clipping to [-0.8416, +0.8416]
- Fixed scaling to signed range [-1, +1]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, class_a_signed

FEATURE_COL = "z_typical_price_30_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, period: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    # Past-only mean and std
    past_tp = tp.shift(1)
    mean = past_tp.rolling(window=period, min_periods=period).mean()
    std = past_tp.rolling(window=period, min_periods=period).std()
    raw = (tp - mean) / (std + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: z_typical_price_30 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=30)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.period, args.norm_window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()