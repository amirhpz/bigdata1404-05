"""
Feature: vwap_dev_atr_20
Class: A (open-domain, signed)

Raw definition:
    (close - vwap20_past) / (ATR14 + eps)

Where (VWAP is past-only):
    vwap20_past_t = sum(close_i * vol_i, i=t-20..t-1) / (sum(vol_i, i=t-20..t-1) + eps)

Normalization policy:
- Robust z-score (past-only, via utility)
- Gaussian central-60% clipping to [-0.8416, +0.8416]
- Fixed scaling to signed range [-1, +1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, atr, class_a_signed


FEATURE_COL = "vwap_dev_atr_20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, vwap_window: int, atr_period: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    # Past-only rolling VWAP to avoid leakage (t uses candles from t-window .. t-1).
    pv = (g["close"] * g["volume"]).shift(1)
    v = g["volume"].shift(1)
    sum_pv = pv.rolling(window=vwap_window, min_periods=vwap_window).sum()
    sum_v = v.rolling(window=vwap_window, min_periods=vwap_window).sum()
    vwap_past = sum_pv / (sum_v + EPS)

    atr_n = atr(g, atr_period)

    raw = (g["close"] - vwap_past) / (atr_n + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: vwap_dev_atr_20 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vwap-window", type=int, default=20)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.vwap_window, args.atr_period, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()