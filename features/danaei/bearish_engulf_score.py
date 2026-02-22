"""
Feature script: bearish_engulf_score

Class: A (Open-domain, shock-sensitive, positive — sparse pattern signal)
Final scale: [0, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Quantifies the intensity of a bearish engulfing pattern.
- Zero unless the candle qualifies as a bearish engulf; positive when it does.
- ATR-scaling makes the score cross-symbol and time-scale invariant.
- Class A because the raw series is sparse (mostly zero) with an open-domain tail.

Raw formula (causal):
    A "bearish engulf" candle t requires (ALL must hold):
        1) prev candle is bullish:      close_{t-1} > open_{t-1}
        2) current candle is bearish:   close_t < open_t
        3) current open engulfs prev:   open_t  >= close_{t-1}  (opens at or above prev close)
        4) current close engulfs prev:  close_t <= open_{t-1}   (closes at or below prev open)

    bearish_engulf_score_raw_t =
        I[bear_engulf_t] * abs(close_t - open_t) / (ATR_14_t + eps)

    where ATR_14_t = EMA(TR, 14) — causal ATR.

Class A pipeline:
1) Compute raw score causally.
2) Rolling robust z-score (past-only median/MAD).
3) Clip to [-0.8416, +0.8416].
4) Scale to [0, 1] (positive-domain).

Notes:
- Computation is per-symbol; no cross-symbol mixing.
- Sparse feature: most rows are zero. Class A handles heavy tail from non-zero events.
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


def atr_ema(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Causal ATR via EMA of True Range (min_periods ensures proper warm-up NaN)."""
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def bearish_engulf_score_raw(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """
    Bearish engulf intensity: I[bear_engulf] * abs(close - open) / (ATR_14 + eps).
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    prev_bullish = prev_close > prev_open
    curr_bearish = df["close"] < df["open"]
    # current opens at or above prev close
    open_engulfs = df["open"] >= prev_close
    # current closes at or below prev open
    close_engulfs = df["close"] <= prev_open

    is_engulf = prev_bullish & curr_bearish & open_engulfs & close_engulfs
    body = (df["close"] - df["open"]).abs()
    atr = atr_ema(df, period=atr_period)

    return is_engulf.astype(float) * body / (atr + EPS)


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


def build_feature_table(
    df: pd.DataFrame, atr_period: int = 14, norm_window: int = 100
) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - bearish_engulf_score_scaled_pos_0_1
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

        raw = bearish_engulf_score_raw(g, atr_period=atr_period)
        final = class_a_pos_0_1(raw, norm_window=norm_window)

        g_out = g.copy()
        g_out["bearish_engulf_score_scaled_pos_0_1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute bearish_engulf_score (Class A) scaled to [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--atr-period", type=int, default=14,
                   help="ATR EMA period (default: 14).")
    p.add_argument("--norm-window", type=int, default=100,
                   help="Robust z-score rolling window (default: 100).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(
        df, atr_period=args.atr_period, norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - bearish_engulf_score_scaled_pos_0_1")


if __name__ == "__main__":
    main()
