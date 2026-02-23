"""
Feature script: channel_pos_20

Class: B (Bounded semantic oscillator / channel position)
Final scale: [0, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Where is the current close located within the prior 20-bar high-low channel?
- 0 = at the bottom of the 20-bar range, 1 = at the top of the 20-bar range.
- Provides a scale-invariant, symbol-agnostic measure of near-term relative position.

Raw formula (causal, past-only):
    high_20_t = max(high_{t-20}..high_{t-1})   (PAST-ONLY via shift(1))
    low_20_t  = min(low_{t-20}..low_{t-1})
    channel_pos_20_t = (close_t - low_20_t) / (high_20_t - low_20_t + eps)

Notes:
- Naturally bounded in [0, 1] by construction; we apply a semantic clip to enforce it.
- No z-score or Gaussian clipping (Class B).
- Rolling is shift(1)-based to avoid leakage.
- Computation is per-symbol; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "channel_pos_20"


def compute_feature(g: pd.DataFrame, window: int = 20) -> pd.Series:
    """Past-only channel position in [0, 1]."""
    past_high = g["high"].shift(1)
    past_low = g["low"].shift(1)
    high_w = past_high.rolling(window=window, min_periods=window).max()
    low_w = past_low.rolling(window=window, min_periods=window).min()
    raw = (g["close"] - low_w) / ((high_w - low_w) + EPS)
    return raw.clip(lower=0.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, window))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute channel_pos_20 (Class B) — past-only 20-bar channel position in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
