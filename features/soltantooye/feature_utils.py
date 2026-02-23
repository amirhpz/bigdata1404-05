"""
Shared utilities for single-feature scripts.

Design intent:
- Keep each feature file small and readable.
- Enforce one consistent processing policy across all features.
- Prevent leakage by default when normalizing Class A features.

Important conventions:
- Class A (open-domain): robust z-score (past-only) + Gaussian 60% clip + fixed scaling.
- Class B (bounded semantics): no z-score, no Gaussian clipping.
- Class C (flags): no normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed Gaussian central-60% bounds.
GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416
EPS = 1e-12


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate minimum required input columns for OHLCV feature computation."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range using current high/low and previous close."""
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR with EMA smoothing and causal warm-up behavior."""
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def rolling_robust_zscore_past_only(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Compute leakage-safe robust z-score.

    Formula:
    rz_t = 0.67448975 * (x_t - median_past) / (MAD_past + eps)

    Leakage prevention:
    - Rolling stats are built on `series.shift(1)` so row t never uses itself.
    """
    past = series.shift(1)
    median = past.rolling(window=window, min_periods=window).median()

    # True rolling MAD:
    # MAD_t = median(|x_i - median_t|) over the same past-only window.
    # We compute it per-window to keep the MAD definition exact.
    def rolling_mad(arr) -> float:
        m = np.median(arr)
        return float(np.median(np.abs(arr - m)))

    mad = past.rolling(window=window, min_periods=window).apply(rolling_mad, raw=True)
    return 0.67448975 * (series - median) / (mad + EPS)


def class_a_signed(raw: pd.Series, norm_window: int) -> pd.Series:
    """
    Class A signed pipeline:
    robust z -> clip to [-0.8416, +0.8416] -> scale to [-1, +1].
    """
    rz = rolling_robust_zscore_past_only(raw, norm_window)
    rz_clip = rz.clip(GAUSSIAN_CLIP_LOW, GAUSSIAN_CLIP_HIGH)
    return rz_clip / GAUSSIAN_CLIP_HIGH


def class_a_positive(raw: pd.Series, norm_window: int) -> pd.Series:
    """
    Class A positive pipeline:
    robust z -> clip to [-0.8416, +0.8416] -> scale to [0, 1].
    """
    rz = rolling_robust_zscore_past_only(raw, norm_window)
    rz_clip = rz.clip(GAUSSIAN_CLIP_LOW, GAUSSIAN_CLIP_HIGH)
    return (rz_clip + GAUSSIAN_CLIP_HIGH) / (2.0 * GAUSSIAN_CLIP_HIGH)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder-style RSI in [0, 100]."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / (avg_down + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def dmi_balance(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute normalized DMI balance:
    (DI+ - DI-) / (DI+ + DI- + eps), approximately in [-1, 1].
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    atr_n = atr(df, period=period)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr_n + EPS)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr_n + EPS)
    return (plus_di - minus_di) / (plus_di + minus_di + EPS)


def add_feature_per_symbol(
    df: pd.DataFrame,
    feature_col: str,
    compute_fn,
) -> pd.DataFrame:
    """
    Append one feature column to the original dataset with symbol-local computation.

    Why symbol-local:
    - Prevent cross-symbol contamination in rolling/statistical logic.
    - Keep each symbol's time series causal and independent.
    """
    validate_ohlcv(df)
    out = df.copy()
    out["_orig_idx"] = np.arange(len(out))

    if "symbol" in out.columns:
        parts = []
        for _, g in out.groupby("symbol", sort=False, group_keys=False):
            if "datetime_utc" in g.columns:
                # Enforce causal order inside each symbol.
                g = g.sort_values("datetime_utc")
            g2 = g.copy()
            g2[feature_col] = compute_fn(g2)
            parts.append(g2)
        merged = pd.concat(parts, axis=0)
    else:
        if "datetime_utc" in out.columns:
            out = out.sort_values("datetime_utc")
        out[feature_col] = compute_fn(out)
        merged = out

    merged = merged.sort_values("_orig_idx").drop(columns=["_orig_idx"])
    return merged
