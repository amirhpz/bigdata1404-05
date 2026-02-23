"""
Feature script: parkinson_vol_20

Class: A (positive-domain)
Raw (past-only rolling mean):
    r_t = ln(high_t / low_t)
    x_t = sqrt( mean(r_{t-1}^2, 20) / (4 ln 2) )

Output column:
- parkinson_vol_20_scaled_pos_0_1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416
GAUSSIAN_CONSISTENCY = 0.67448975
EPS = 1e-12


def rolling_mean_past_only(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=window).mean()


def raw_parkinson_vol_20(df: pd.DataFrame) -> pd.Series:
    r = np.log((df["high"] + EPS) / (df["low"] + EPS))
    r2_mean = rolling_mean_past_only(r * r, window=20)
    return np.sqrt(r2_mean / (4.0 * np.log(2.0) + EPS))


def rolling_robust_zscore_past_only(series: pd.Series, window: int = 100) -> pd.Series:
    past = series.shift(1)
    median = past.rolling(window=window, min_periods=window).median()
    mad = (past - median).abs().rolling(window=window, min_periods=window).median()
    return GAUSSIAN_CONSISTENCY * (series - median) / (mad + EPS)


def class_a_pos_0_1(raw: pd.Series, norm_window: int = 100) -> pd.Series:
    rz = rolling_robust_zscore_past_only(raw, window=norm_window)
    rz_clip = rz.clip(GAUSSIAN_CLIP_LOW, GAUSSIAN_CLIP_HIGH)
    return (rz_clip + GAUSSIAN_CLIP_HIGH) / (2 * GAUSSIAN_CLIP_HIGH)


def build_feature_table(df: pd.DataFrame, norm_window: int = 100) -> pd.DataFrame:
    required_cols = {"high", "low"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    grouped = out.groupby("symbol", group_keys=False, sort=False) if "symbol" in out.columns else [(None, out)]
    parts = []

    for _, g in grouped:
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        raw = raw_parkinson_vol_20(g)
        final = class_a_pos_0_1(raw, norm_window=norm_window)

        g_out = g.copy()
        g_out["parkinson_vol_20_scaled_pos_0_1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute parkinson_vol_20 feature (Class A) scaled to [0,1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-window", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}\nColumns added:\n - parkinson_vol_20_scaled_pos_0_1")


if __name__ == "__main__":
    main()