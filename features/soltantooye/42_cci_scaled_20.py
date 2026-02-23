"""
Feature: cci_scaled_20
Class: A (open-domain, signed)

Sheet definition:
    CCI(20) / 200

CCI formula:
    TP = (H + L + C)/3
    SMA_TP = SMA(TP, 20)
    MeanDev = mean(|TP - SMA_TP|, 20)
    CCI = (TP - SMA_TP) / (0.015 * MeanDev + eps)

Raw definition used:
    raw = CCI20 / 200

Normalization policy:
- Robust z-score (past-only) + Gaussian clip + scale to [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_utils import EPS, add_feature_per_symbol, class_a_signed


FEATURE_COL = "cci_scaled_20_scaled_signed_m1_p1"


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(window=period, min_periods=period).mean()

    # Mean absolute deviation around SMA (causal; uses current in-window, not future).
    # This is part of the indicator definition, not a normalization-fit step.
    mad = (tp - sma).abs().rolling(window=period, min_periods=period).mean()

    return (tp - sma) / (0.015 * mad + EPS)


def compute_feature(g: pd.DataFrame, cci_period: int, norm_window: int) -> pd.Series:
    raw = cci(g, cci_period) / 200.0
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: cci_scaled_20 (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cci-period", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.cci_period, args.norm_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()