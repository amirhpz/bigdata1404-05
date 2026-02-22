"""
Feature: open_gap_atr
Class: A (open-domain, signed)
Raw definition:
    Gap magnitude vs normal range: (O-Cl[t-1])/ATR_14
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

FEATURE_COL = "open_gap_atr_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, atr_period: int, norm_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final scaled column."""
    prev_close = g["close"].shift(1)
    gap = g["open"] - prev_close
    raw = gap / (atr(g, atr_period) + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: open_gap_atr (Class A signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.atr_period, args.norm_window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()