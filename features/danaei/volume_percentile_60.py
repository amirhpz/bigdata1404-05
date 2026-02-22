"""
Feature script: volume_percentile_60

Class: B (Bounded semantic — percentile rank)
Final scale: [0, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Rank of the current volume among the past 60 bars.
- A percentile rank is completely scale-free and distribution-agnostic.
- 0 = lowest volume in last 60 bars, 1 = highest volume in last 60 bars.
- Captures relative participation without sensitivity to absolute volume levels.

Raw formula (causal, past-only):
    volume_percentile_60_t =
        count(volume_{t-60}..volume_{t-1} <= volume_t) / 60

Notes:
- Past-only: current volume is ranked against the PREVIOUS 60 bars.
- Naturally bounded in [0, 1]; no further scaling needed (Class B).
- No z-score or Gaussian clipping.
- Computation is per-symbol; no cross-symbol mixing.
- Early rows NaN until 60 past bars are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def volume_percentile_60(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Past-only volume percentile rank in [0, 1].

    percentile_t = fraction of past `window` volumes <= current volume.
    """
    vol = df["volume"].values
    n = len(vol)
    result = np.full(n, np.nan)

    for i in range(window, n):
        # exactly the prior `window` values
        past = vol[i - window: i]
        current = vol[i]
        result[i] = np.sum(past <= current) / window

    return pd.Series(result, index=df.index)


def build_feature_table(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - volume_percentile_60  (bounded semantic in [0, 1])
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
        g_out["volume_percentile_60"] = volume_percentile_60(g, window=window)
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute volume_percentile_60 (Class B) — past-only volume percentile rank in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--window", type=int, default=60,
                   help="Lookback window for percentile rank (default: 60).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - volume_percentile_60")


if __name__ == "__main__":
    main()
