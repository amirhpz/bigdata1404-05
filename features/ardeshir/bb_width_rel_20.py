# =========================
# File: features/bb_width_rel_20.py
# =========================
"""
Feature: bb_width_rel_20
Class: A (open-domain, positive)

Raw definition (past-only):
    (BB_upper - BB_lower) / EMA20
Equivalent:
    (2 * num_std * rolling_std_20(past_close)) / EMA20(past_close)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features.feature_utils import EPS, add_feature_per_symbol, class_a_positive


FEATURE_COL = "bb_width_rel_20_scaled_pos_0_1"


def compute_feature(
    g: pd.DataFrame,
    period: int,
    num_std: float,
    norm_window: int,
) -> pd.Series:
    """Compute normalized Bollinger relative width for one symbol slice (past-only)."""
    close = g["close"]
    past_close = close.shift(1)
    ema = past_close.ewm(span=period, adjust=False, min_periods=period).mean()
    std = past_close.rolling(window=period, min_periods=period).std(ddof=0)
    raw = (2.0 * num_std * std) / (ema + EPS)
    return class_a_positive(raw, norm_window)


def build_feature_table(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    norm_window: int = 100,
) -> pd.DataFrame:
    return add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, period, num_std, norm_window),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: bb_width_rel_20 (Class A positive)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=20)
    p.add_argument("--num-std", type=float, default=2.0)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(df, period=args.period, num_std=args.num_std, norm_window=args.norm_window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()