"""
Feature script: atr_pct_14

Class: A (positive-domain)
Raw (relative, causal where rolling is used):
    ATR14_past_ema / close_{t-1}

Pipeline:
1) Compute raw feature per symbol (past-only ATR).
2) Robust z-score (past-only median/MAD).
3) Clip to [-0.8416, +0.8416].
4) Scale to [0,1] (positive-domain).

Output column:
- atr_pct_14_scaled_pos_0_1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416
GAUSSIAN_CONSISTENCY = 0.67448975
EPS = 1e-12


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr_ema_past_only(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # ATR_t = EMA(TR_{t-1}, period)
    tr = true_range(df)
    tr_past = tr.shift(1)
    return tr_past.ewm(span=period, adjust=False, min_periods=period).mean()


def raw_atr_pct_14(df: pd.DataFrame) -> pd.Series:
    atr14 = atr_ema_past_only(df, period=14)
    close_past = df["close"].shift(1)
    return atr14 / (close_past + EPS)


def rolling_robust_zscore_past_only(series: pd.Series, window: int = 100) -> pd.Series:
    past = series.shift(1)
    median = past.rolling(window=window, min_periods=window).median()
    abs_dev = (past - median).abs()
    mad = abs_dev.rolling(window=window, min_periods=window).median()
    return GAUSSIAN_CONSISTENCY * (series - median) / (mad + EPS)


def class_a_pos_0_1(series_raw: pd.Series, norm_window: int = 100) -> pd.Series:
    rz = rolling_robust_zscore_past_only(series_raw, window=norm_window)
    rz_clip = rz.clip(lower=GAUSSIAN_CLIP_LOW, upper=GAUSSIAN_CLIP_HIGH)
    return (rz_clip + GAUSSIAN_CLIP_HIGH) / (2 * GAUSSIAN_CLIP_HIGH)


def build_feature_table(df: pd.DataFrame, norm_window: int = 100) -> pd.DataFrame:
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    grouped = out.groupby("symbol", group_keys=False, sort=False) if "symbol" in out.columns else [(None, out)]

    parts = []
    for _, g in grouped:
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        raw = raw_atr_pct_14(g)
        final = class_a_pos_0_1(raw, norm_window=norm_window)

        g_out = g.copy()
        g_out["atr_pct_14_scaled_pos_0_1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute atr_pct_14 feature (Class A) scaled to [0,1].")
    p.add_argument("--input", type=Path, required=True, help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    p.add_argument("--norm-window", type=int, default=100, help="Robust z-score rolling window (default: 100).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - atr_pct_14_scaled_pos_0_1")


if __name__ == "__main__":
    main()