"""
Feature script: range_compression_20_100

Class: B (Bounded semantic oscillator / range ratio)
Final scale: [0, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Measures near-term compression relative to a longer-term range.
- Low value (near 0): price is highly compressed over the recent 20 bars vs the 100-bar range.
- High value (near 1): recent 20-bar range nearly spans the 100-bar range (expansion).

Raw formula (causal, past-only):
    high_20_t  = max(high_{t-20}..high_{t-1})
    low_20_t   = min(low_{t-20}..low_{t-1})
    high_100_t = max(high_{t-100}..high_{t-1})
    low_100_t  = min(low_{t-100}..low_{t-1})

    range_compression_20_100_t = (high_20_t - low_20_t) / (high_100_t - low_100_t + eps)

Notes:
- Naturally bounded in [0, 1] because the 20-bar sub-range cannot exceed the 100-bar range.
- Semantic clip to [0, 1] enforced to handle numerical edge cases.
- No z-score or Gaussian clipping (Class B).
- Rolling is shift(1)-based to avoid leakage.
- Computation is per-symbol; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EPS = 1e-12


def range_compression_20_100(df: pd.DataFrame, short_window: int = 20, long_window: int = 100) -> pd.Series:
    """
    Past-only range compression ratio in [0, 1].

    range_compression_t = (high_short_t - low_short_t) / (high_long_t - low_long_t + eps)
    """
    past_high = df["high"].shift(1)
    past_low = df["low"].shift(1)

    high_short = past_high.rolling(
        window=short_window, min_periods=short_window).max()
    low_short = past_low.rolling(
        window=short_window, min_periods=short_window).min()

    high_long = past_high.rolling(
        window=long_window, min_periods=long_window).max()
    low_long = past_low.rolling(
        window=long_window, min_periods=long_window).min()

    raw = (high_short - low_short) / ((high_long - low_long) + EPS)
    return raw.clip(lower=0.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, short_window: int = 20, long_window: int = 100) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - range_compression_20_100  (bounded semantic in [0, 1])
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
        g_out["range_compression_20_100"] = range_compression_20_100(
            g, short_window=short_window, long_window=long_window
        )
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute range_compression_20_100 (Class B) — range ratio in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--short-window", type=int, default=20,
                   help="Short lookback window (default: 20).")
    p.add_argument("--long-window", type=int, default=100,
                   help="Long lookback window (default: 100).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(
        df, short_window=args.short_window, long_window=args.long_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - range_compression_20_100")


if __name__ == "__main__":
    main()
