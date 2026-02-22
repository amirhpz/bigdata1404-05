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

GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416
GAUSSIAN_CONSISTENCY = 0.67448975
EPS = 1e-12


def vol_over_median20_raw(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Volume relative to its past-only rolling median.

    vol_over_median20_t = volume_t / (median(volume_{t-window}..volume_{t-1}) + eps)
    """
    past_vol = df["volume"].shift(1)
    median_v = past_vol.rolling(window=window, min_periods=window).median()
    return df["volume"] / (median_v + EPS)


def rolling_robust_zscore_past_only(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Past-only rolling robust z-score:
      rz_t = 0.67448975 * (x_t - median_past) / (MAD_past + eps)
    """
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


def build_feature_table(df: pd.DataFrame, window: int = 20, norm_window: int = 100) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - vol_over_median20_scaled_pos_0_1
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

        raw = vol_over_median20_raw(g, window=window)
        final = class_a_pos_0_1(raw, norm_window=norm_window)

        g_out = g.copy()
        g_out["vol_over_median20_scaled_pos_0_1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute vol_over_median20 (Class A) scaled to [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--window", type=int, default=20,
                   help="Median lookback window (default: 20).")
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
    print("Columns added:\n - vol_over_median20_scaled_pos_0_1")


if __name__ == "__main__":
    main()
