"""
Feature script: percent_b_20

Class: B (Bounded semantic oscillator / Bollinger Band position)
Final scale: [-1, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Measures where the current close sits within the 20-bar Bollinger Band.
- %B = 0 means price is at lower band, %B = 1 means price is at upper band.
- Transformed to [-1, 1] via the standard Class B semantic: 2*(%B - 0.5).
- Useful for mean-reversion signals: -1 = deeply oversold, +1 = deeply overbought.

Raw formula (causal, past-only):
    SMA_20_t   = mean(close_{t-20}..close_{t-1})     (PAST-ONLY via shift(1))
    std_20_t   = std(close_{t-20}..close_{t-1})
    BB_up_20_t = SMA_20_t + 2 * std_20_t
    BB_dn_20_t = SMA_20_t - 2 * std_20_t

    pct_b_raw_t = (close_t - BB_dn_20_t) / (BB_up_20_t - BB_dn_20_t + eps)  ∈ [0, 1]

    percent_b_20_t = 2 * (pct_b_raw_t - 0.5)   ∈ [-1, 1]

Notes:
- No z-score or Gaussian clipping (Class B).
- Mild semantic clip to [-1, 1] enforced to handle rare extreme excursions beyond bands.
- Rolling stats are shift(1)-based (past-only) to avoid leakage.
- Computation is per-symbol; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EPS = 1e-12
BB_MULTIPLIER = 2.0


def bollinger_bands_past_only(df: pd.DataFrame, window: int = 20, num_std: float = BB_MULTIPLIER):
    """
    Past-only Bollinger Bands (SMA +/- num_std * sigma).

    Returns (bb_up, bb_dn) as pd.Series pair.
    """
    past_close = df["close"].shift(1)
    sma = past_close.rolling(window=window, min_periods=window).mean()
    std = past_close.rolling(window=window, min_periods=window).std(ddof=1)
    bb_up = sma + num_std * std
    bb_dn = sma - num_std * std
    return bb_up, bb_dn


def percent_b_20(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Bollinger Band %B transformed to [-1, 1].

    pct_b_raw  = (close - BB_dn) / (BB_up - BB_dn + eps)     ∈ [0, 1]
    percent_b  = 2 * (pct_b_raw - 0.5)                       ∈ [-1, 1]
    """
    bb_up, bb_dn = bollinger_bands_past_only(df, window=window)
    pct_b_raw = (df["close"] - bb_dn) / ((bb_up - bb_dn) + EPS)
    raw_signed = 2.0 * (pct_b_raw - 0.5)
    return raw_signed.clip(lower=-1.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - percent_b_20  (bounded semantic in [-1, 1])
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

        g_out = g.copy()
        g_out["percent_b_20"] = percent_b_20(g, window=window)
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute percent_b_20 (Class B) — Bollinger Band %B transformed to [-1, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--window", type=int, default=20,
                   help="Bollinger Band lookback (default: 20).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - percent_b_20")


if __name__ == "__main__":
    main()
