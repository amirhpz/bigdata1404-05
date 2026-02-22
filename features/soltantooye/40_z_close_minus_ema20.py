"""
Feature: z_close_minus_ema20
Class: A (open-domain, signed)
Raw definition:
    Deviation from mean trend: (C-EMA(20))/(std(C,20)+eps)
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

FEATURE_COL = "z_close_minus_ema20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, ema_period: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    ema = g["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    # Rolling z-score using past-only std
    past_close = g["close"].shift(1)
    std = past_close.rolling(window=ema_period, min_periods=ema_period).std()
    raw = (g["close"] - ema) / (std + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: z_close_minus_ema20 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ema-period", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.ema_period, args.norm_window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
    