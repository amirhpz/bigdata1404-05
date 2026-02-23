# =========================
# File: features/vol_of_vol_ratio_14_20.py
# =========================
"""
Feature: vol_of_vol_ratio_14_20
Class: A (open-domain, positive)

Raw definition (past-only components):
    y_t = TR_t / ATR14_t
    vol_of_vol_t = STD20(y_{t-1}) / EMA20(y_{t-1})

Note:
- TR_t uses current candle by definition (high/low/prev_close).
- The rolling/std/ema used to summarize y are computed on y.shift(1) (past-only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features.feature_utils import EPS, add_feature_per_symbol, class_a_positive, true_range, atr


FEATURE_COL = "vol_of_vol_ratio_14_20_scaled_pos_0_1"


def compute_feature(
    g: pd.DataFrame,
    atr_period: int,
    period: int,
    norm_window: int,
) -> pd.Series:
    """Compute vol-of-vol ratio for one symbol slice (past-only stats)."""
    tr = true_range(g)
    atr_n = atr(g, period=atr_period)
    y = tr / (atr_n + EPS)

    past_y = y.shift(1)
    std = past_y.rolling(window=period, min_periods=period).std(ddof=0)
    ema = past_y.ewm(span=period, adjust=False, min_periods=period).mean()

    raw = std / (ema + EPS)
    return class_a_positive(raw, norm_window)


def build_feature_table(
    df: pd.DataFrame,
    atr_period: int = 14,
    period: int = 20,
    norm_window: int = 100,
) -> pd.DataFrame:
    return add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, atr_period, period, norm_window),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: vol_of_vol_ratio_14_20 (Class A positive)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--period", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, atr_period=args.atr_period, period=args.period, norm_window=args.norm_window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()