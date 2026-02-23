"""
Feature script: log_range_over_vol_100

Class: A (Open-domain, shock-sensitive, positive)
Final scale: [0, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Directional persistence proxy: how much has price moved net vs how volatile it has been.
- High value: log-price range (peak-to-trough) is large relative to return volatility
  => price has trended strongly in one direction.
- Low value: price has been choppy — lots of volatility but little net movement.

Raw formula (causal, past-only):
    log_close_t = log(close_t)
    ret1_t      = log_close_t - log_close_{t-1}

    log_range_100_t = max(log_close_{t-100}..log_close_{t-1})
                    - min(log_close_{t-100}..log_close_{t-1})

    ret_vol_100_t   = std(ret1_{t-100}..ret1_{t-1})   (past-only, ddof=1)

    log_range_over_vol_100_raw_t = log_range_100_t / (ret_vol_100_t + eps)

Class A pipeline:
1) Compute raw ratio causally.
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

import numpy as np
import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, class_a_positive

FEATURE_COL = "log_range_over_vol_100_scaled_pos_0_1"


def compute_feature(g: pd.DataFrame, window: int = 100, norm_window: int = 100) -> pd.Series:
    """Log-price range over return vol, normalized via Class A pipeline."""
    log_c = np.log(g["close"] + EPS)
    ret1 = log_c.diff(1)
    past_log_c = log_c.shift(1)
    past_ret1 = ret1.shift(1)
    max_log = past_log_c.rolling(window=window, min_periods=window).max()
    min_log = past_log_c.rolling(window=window, min_periods=window).min()
    log_range = max_log - min_log
    ret_vol = past_ret1.rolling(window=window, min_periods=window).std(ddof=1)
    raw = log_range / (ret_vol + EPS)
    return class_a_positive(raw, norm_window)


def build_feature_table(df: pd.DataFrame, window: int = 100, norm_window: int = 100) -> pd.DataFrame:
    return add_feature_per_symbol(
        df, FEATURE_COL, lambda g: compute_feature(g, window, norm_window)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute log_range_over_vol_100 (Class A) scaled to [0, 1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=100)
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
