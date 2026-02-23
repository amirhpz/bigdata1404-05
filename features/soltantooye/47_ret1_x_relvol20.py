"""
Feature: ret1_x_relvol20
Class: A (open-domain, signed)

Sheet definition:
    ret1 * (V / (EMA(V,20) + eps))

Implementation (leakage-safe relative-volume baseline):
- ret1 = close/close[-1] - 1
- relvol = volume / (EMA(volume.shift(1), 20) + eps)
  (EMA baseline uses only past volumes up to t-1; current volume stays in numerator)

Normalization policy (Class A):
- Robust z-score (past-only, per-symbol)
- Gaussian central-60% clipping to [-0.8416, +0.8416]
- Fixed scaling to signed range [-1, +1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_utils import EPS, add_feature_per_symbol, class_a_signed


FEATURE_COL = "ret1_x_relvol20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, ema_period: int, norm_window: int) -> pd.Series:
    ret1 = g["close"] / g["close"].shift(1) - 1.0

    ema_v = (
        g["volume"]
        .shift(1)
        .ewm(span=ema_period, adjust=False, min_periods=ema_period)
        .mean()
    )
    relvol = g["volume"] / (ema_v + EPS)

    raw = ret1 * relvol
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: ret1_x_relvol20 (Class A signed)")
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