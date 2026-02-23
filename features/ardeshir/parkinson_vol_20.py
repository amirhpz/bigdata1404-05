# =========================
# File: features/parkinson_vol_20.py
# =========================
"""
Feature: parkinson_vol_20
Class: A (open-domain, positive)

Raw definition (past-only):
    r_t = ln(high_t / low_t)
    parkinson_t = sqrt( mean(r_{t-1}^2, 20) / (4 ln 2) )
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features.feature_utils import EPS, add_feature_per_symbol, class_a_positive


FEATURE_COL = "parkinson_vol_20_scaled_pos_0_1"


def compute_feature(
    g: pd.DataFrame,
    period: int,
    norm_window: int,
) -> pd.Series:
    """Compute Parkinson volatility for one symbol slice (past-only)."""
    r = np.log((g["high"] + EPS) / (g["low"] + EPS))
    past_r = r.shift(1)
    mean_r2 = (past_r * past_r).rolling(window=period, min_periods=period).mean()
    raw = np.sqrt(mean_r2 / (4.0 * np.log(2.0) + EPS))
    return class_a_positive(raw, norm_window)


def build_feature_table(
    df: pd.DataFrame,
    period: int = 20,
    norm_window: int = 100,
) -> pd.DataFrame:
    return add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, period, norm_window),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: parkinson_vol_20 (Class A positive)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, period=args.period, norm_window=args.norm_window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()