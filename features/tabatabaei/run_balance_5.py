"""
Feature: run_balance_5
Class: B (bounded semantic, signed)

Raw definition (past-only run balance):
    (count(up, 5) - count(down, 5)) / 5

Where:
    ret1_t = close_t / close_{t-1} - 1
    up_t   = 1(ret1_t > 0)
    down_t = 1(ret1_t < 0)

Counts are computed over:
    t-5 .. t-1  (past-only)

Processing policy (Class B):
- No z-score / robust z-score
- No Gaussian clipping
- Semantic bounded output in [-1, +1] (apply strict safety clip)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol


FEATURE_COL = "run_balance_5"


def compute_feature(g: pd.DataFrame, lookback: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final bounded column."""
    prev_close = g["close"].shift(1)
    ret1 = (g["close"] / (prev_close + EPS)) - 1.0

    up = (ret1 > 0).astype("float64")
    down = (ret1 < 0).astype("float64")

    # Past-only rolling counts to avoid leakage.
    up_cnt = up.shift(1).rolling(window=lookback, min_periods=lookback).sum()
    down_cnt = down.shift(1).rolling(window=lookback, min_periods=lookback).sum()

    raw = (up_cnt - down_cnt) / float(lookback)
    return raw.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: run_balance_5 (Class B signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.lookback),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()