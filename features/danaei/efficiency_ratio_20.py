"""
Feature script: efficiency_ratio_20

Class: B (Bounded semantic — Kaufman Efficiency Ratio)
Final scale: [0, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Measures how efficiently price has moved over the past 20 bars.
- ER = net price change / total path length travelled.
- 0 = perfectly choppy (all movement reversed), 1 = perfectly directional (straight line).
- Scale-invariant because both numerator and denominator are in price units (ratio).

Raw formula (causal, past-only):
    net_move_t   = abs(close_t - close_{t-20})             (net displacement)
    path_t       = sum(abs(close_{t-k} - close_{t-k-1}),
                        k=0..19)                           (sum of 1-bar abs moves over past 20 bars)
    efficiency_ratio_20_t = net_move_t / (path_t + eps)

Anti-leakage:
- close_{t-20} and close_{t-1} are both past values. No future data used.
- path_t uses shifts entirely within past 20 candles.

Notes:
- Naturally bounded in [0, 1]; semantic clip applied as a safeguard.
- No z-score or Gaussian clipping (Class B).
- Computation is per-symbol; no cross-symbol mixing.
- Early rows NaN until 21 candles of history are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "efficiency_ratio_20"


def compute_feature(g: pd.DataFrame, window: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio over past `window` bars, in [0, 1]."""
    close = g["close"]
    net_move = (close - close.shift(window)).abs()
    path_len = close.diff(1).abs().rolling(
        window=window, min_periods=window).sum()
    raw = net_move / (path_len + EPS)
    return raw.clip(lower=0.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, window))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute efficiency_ratio_20 (Class B) — Kaufman ER in [0, 1]."
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
