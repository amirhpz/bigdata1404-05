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

GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416
GAUSSIAN_CONSISTENCY = 0.67448975
EPS = 1e-12


def log_range_over_vol_100_raw(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Log-price range over past `window` bars divided by past return volatility.

    Raw_t = (max_logC_past - min_logC_past) / (std(ret1_past) + eps)
    """
    log_c = np.log(df["close"] + EPS)
    ret1 = log_c.diff(1)        # log return, causal difference

    past_log_c = log_c.shift(1)
    past_ret1 = ret1.shift(1)

    max_log = past_log_c.rolling(window=window, min_periods=window).max()
    min_log = past_log_c.rolling(window=window, min_periods=window).min()
    log_range = max_log - min_log

    ret_vol = past_ret1.rolling(window=window, min_periods=window).std(ddof=1)

    return log_range / (ret_vol + EPS)


def rolling_robust_zscore_past_only(series: pd.Series, window: int = 100) -> pd.Series:
    """Past-only rolling robust z-score."""
    past = series.shift(1)
    median = past.rolling(window=window, min_periods=window).median()
    abs_dev = (past - median).abs()
    mad = abs_dev.rolling(window=window, min_periods=window).median()
    return GAUSSIAN_CONSISTENCY * (series - median) / (mad + EPS)


def class_a_pos_0_1(series_raw: pd.Series, norm_window: int = 100) -> pd.Series:
    """Class A pipeline: robust z -> clip -> scale to [0, 1]."""
    rz = rolling_robust_zscore_past_only(series_raw, window=norm_window)
    rz_clip = rz.clip(lower=GAUSSIAN_CLIP_LOW, upper=GAUSSIAN_CLIP_HIGH)
    return (rz_clip + GAUSSIAN_CLIP_HIGH) / (2.0 * GAUSSIAN_CLIP_HIGH)


def build_feature_table(df: pd.DataFrame, window: int = 100, norm_window: int = 100) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - log_range_over_vol_100_scaled_pos_0_1
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    if "symbol" in out.columns:
        grouped = out.groupby("symbol", group_keys=False, sort=False)
    else:
        grouped = [(None, out)]

    parts = []
    for _, g in grouped:
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        raw = log_range_over_vol_100_raw(g, window=window)
        final = class_a_pos_0_1(raw, norm_window=norm_window)

        g_out = g.copy()
        g_out["log_range_over_vol_100_scaled_pos_0_1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute log_range_over_vol_100 (Class A) scaled to [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--window", type=int, default=100,
                   help="Lookback window (default: 100).")
    p.add_argument("--norm-window", type=int, default=100,
                   help="Robust z-score rolling window (default: 100).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window,
                              norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - log_range_over_vol_100_scaled_pos_0_1")


if __name__ == "__main__":
    main()
