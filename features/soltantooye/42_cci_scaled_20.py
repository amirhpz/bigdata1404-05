"""
Feature: cci_scaled_20
Class: A (open-domain, signed)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, class_a_signed

FEATURE_COL = "cci_scaled_20_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, period: int, norm_window: int) -> pd.Series:
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    past_tp = tp.shift(1)
    cl = past_tp.rolling(window=period, min_periods=period).mean()
    sd = past_tp.rolling(window=period, min_periods=period).std()
    raw = (g["close"] - cl) / (sd + EPS)
    return class_a_signed(raw, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: cci_scaled_20")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=20)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df, FEATURE_COL,
        lambda g: compute_feature(g, args.period, args.norm_window),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Column: {FEATURE_COL}")


if __name__ == "__main__":
    main()