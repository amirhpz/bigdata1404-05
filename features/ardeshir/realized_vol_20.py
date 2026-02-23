# =========================
# File: features/realized_vol_20.py
# =========================
"""
Feature: realized_vol_20
Class: A (open-domain, positive)

Raw definition (past-only):
    ret1_t = ln(close_t / close_{t-1})
    realized_vol_t = rolling_std_20(ret1_{t-1})   # past-only window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features.feature_utils import EPS, add_feature_per_symbol, class_a_positive


FEATURE_COL = "realized_vol_20_scaled_pos_0_1"


def compute_feature(
    g: pd.DataFrame,
    period: int,
    norm_window: int,
) -> pd.Series:
    """Compute realized volatility for one symbol slice (past-only)."""
    close = g["close"]
    ret1 = np.log((close + EPS) / (close.shift(1) + EPS))
    past_ret1 = ret1.shift(1)
    raw = past_ret1.rolling(window=period, min_periods=period).std(ddof=0)
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
    p = argparse.ArgumentParser(description="Feature: realized_vol_20 (Class A positive)")
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