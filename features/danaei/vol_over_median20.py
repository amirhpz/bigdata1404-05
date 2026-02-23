"""
Feature script: vol_over_median20

Class: A (Open-domain, shock-sensitive, positive)
Final scale: [0, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Measures current volume relative to its recent median baseline.
- Raw value > 1 means above-average participation; < 1 means below.
- Class A because volume spikes can be extreme and distribution is right-skewed.

Raw formula (causal, past-only):
    median_v_t = rolling_median(volume, 20, past-only)
    vol_over_median20_raw_t = volume_t / (median_v_t + eps)

Class A pipeline:
1) Compute raw ratio causally (past-only median via shift(1)).
2) Rolling robust z-score (past-only median/MAD).
3) Clip to [-0.8416, +0.8416].
4) Scale to [0, 1] (positive-domain).

Notes:
- Computation is per-symbol; no cross-symbol mixing.
- Early rows NaN due to rolling warm-up (expected).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, class_a_positive

FEATURE_COL = "vol_over_median20_scaled_pos_0_1"


def compute_feature(g: pd.DataFrame, window: int = 20, norm_window: int = 100) -> pd.Series:
    """Volume over past-only rolling median, normalized via Class A pipeline."""
    past_vol = g["volume"].shift(1)
    median_v = past_vol.rolling(window=window, min_periods=window).median()
    raw = g["volume"] / (median_v + EPS)
    return class_a_positive(raw, norm_window)


def build_feature_table(df: pd.DataFrame, window: int = 20, norm_window: int = 100) -> pd.DataFrame:
    return add_feature_per_symbol(
        df, FEATURE_COL, lambda g: compute_feature(g, window, norm_window)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute vol_over_median20 (Class A) scaled to [0, 1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window,
                              norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
