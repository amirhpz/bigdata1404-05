"""
Feature script: vol_over_ema20

Class: A (Open-domain, shock-sensitive, positive)
Final scale: [0, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Measures current volume relative to its smoothed EMA baseline.
- EMA smoothing reduces the impact of single outlier candles more than a raw median.
- Class A because the ratio is open-domain with heavy right tail.

Raw formula (causal, past-only):
    EMA_V_t    = EMA(volume, 20) computed causally (standard EMA, includes current)
    vol_over_ema20_raw_t = volume_t / (EMA_V_t + eps)

    Note: EMA(span=20) is inherently causal — each value only depends on past data.
    We do NOT shift EMA before dividing, as the EMA at time t already excludes future t+1.

Class A pipeline:
1) Compute raw ratio causally (EMA is naturally past-weighted).
2) Rolling robust z-score (past-only median/MAD).
3) Clip to [-0.8416, +0.8416].
4) Scale to [0, 1] (positive-domain).

Notes:
- Computation is per-symbol; no cross-symbol mixing.
- Early rows NaN due to EMA and normalization warm-up (expected).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, class_a_positive

FEATURE_COL = "vol_over_ema20_scaled_pos_0_1"


def compute_feature(g: pd.DataFrame, span: int = 20, norm_window: int = 100) -> pd.Series:
    """Volume over causal EMA, normalized via Class A pipeline."""
    ema_v = g["volume"].ewm(span=span, adjust=False, min_periods=span).mean()
    raw = g["volume"] / (ema_v + EPS)
    return class_a_positive(raw, norm_window)


def build_feature_table(df: pd.DataFrame, span: int = 20, norm_window: int = 100) -> pd.DataFrame:
    return add_feature_per_symbol(
        df, FEATURE_COL, lambda g: compute_feature(g, span, norm_window)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute vol_over_ema20 (Class A) scaled to [0, 1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--span", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, span=args.span, norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
