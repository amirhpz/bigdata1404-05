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
from feature_utils import add_feature_per_symbol

FEATURE_COL = "volume_percentile_60"


def compute_feature(g: pd.DataFrame, window: int = 60) -> pd.Series:
    """Past-only volume percentile rank in [0, 1]."""
    vol = g["volume"].values
    n = len(vol)
    result = np.full(n, np.nan)
    for i in range(window, n):
        past = vol[i - window: i]
        result[i] = np.sum(past <= vol[i]) / window
    return pd.Series(result, index=g.index)


def build_feature_table(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, window))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute volume_percentile_60 (Class B) — past-only volume percentile rank in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=60)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
