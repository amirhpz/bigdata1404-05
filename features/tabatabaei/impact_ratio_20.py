"""
Feature: impact_ratio_20
Class: A (open-domain, signed)

Raw definition:
    ret1 / ( (volume / (EMA(volume, 20) + eps)) + eps )

Where:
    ret1_t = close_t / close_{t-1} - 1

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


FEATURE_COL = "impact_ratio_20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, vol_ema_span: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    prev_close = g["close"].shift(1)
    ret1 = (g["close"] / (prev_close + EPS)) - 1.0

    vol_ema = g["volume"].ewm(span=vol_ema_span, adjust=False, min_periods=vol_ema_span).mean()
    rel_vol = g["volume"] / (vol_ema + EPS)

    raw = ret1 / (rel_vol + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: impact_ratio_20 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vol-ema-span", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.vol_ema_span, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()