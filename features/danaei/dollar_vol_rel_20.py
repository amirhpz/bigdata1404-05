"""
Feature script: dollar_vol_rel_20

Class: A (Open-domain, shock-sensitive, positive)
Final scale: [0, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Measures the traded dollar value intensity relative to its smoothed EMA baseline.
- Dollar volume (V * C) captures both volume and price magnitude.
- Dividing by its own EMA makes the feature relative and cross-symbol comparable.
- Class A because dollar-volume spikes occur during high-impact events and are open-domain.

Raw formula (causal, past-only):
    dv_t = volume_t * close_t
    EMA_dv_t = EMA(dv, 20) — causal EMA
    dollar_vol_rel_20_raw_t = dv_t / (EMA_dv_t + eps)

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

FEATURE_COL = "dollar_vol_rel_20_scaled_pos_0_1"


def compute_feature(g: pd.DataFrame, span: int = 20, norm_window: int = 100) -> pd.Series:
    """Dollar-volume over causal EMA of dollar-volume, normalized via Class A pipeline."""
    dv = g["volume"] * g["close"]
    ema_dv = dv.ewm(span=span, adjust=False, min_periods=span).mean()
    raw = dv / (ema_dv + EPS)
    return class_a_positive(raw, norm_window)


def build_feature_table(df: pd.DataFrame, span: int = 20, norm_window: int = 100) -> pd.DataFrame:
    return add_feature_per_symbol(
        df, FEATURE_COL, lambda g: compute_feature(g, span, norm_window)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute dollar_vol_rel_20 (Class A) scaled to [0, 1].")
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
