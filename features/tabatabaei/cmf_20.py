"""
Feature: cmf_20
Class: B (bounded semantic, signed)

Raw definition (past-only CMF):
    MFM_t = (2*close_t - high_t - low_t) / (high_t - low_t + eps)
    MFV_t = MFM_t * volume_t
    CMF_t = sum(MFV_{t-20}..MFV_{t-1}) / (sum(volume_{t-20}..volume_{t-1}) + eps)

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


FEATURE_COL = "cmf_20"


def compute_feature(g: pd.DataFrame, lookback: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final bounded column."""
    # Money Flow Multiplier (approx in [-1, 1])
    mfm = (2.0 * g["close"] - g["high"] - g["low"]) / ((g["high"] - g["low"]).abs() + EPS)
    mfv = mfm * g["volume"]

    # Past-only rolling sums to prevent leakage.
    sum_mfv = mfv.shift(1).rolling(window=lookback, min_periods=lookback).sum()
    sum_vol = g["volume"].shift(1).rolling(window=lookback, min_periods=lookback).sum()

    raw = sum_mfv / (sum_vol + EPS)
    return raw.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: cmf_20 (Class B signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=20)
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