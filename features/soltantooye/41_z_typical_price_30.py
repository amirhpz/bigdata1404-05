"""
Feature: z_typical_price_30
Class: A (open-domain, signed) [Robust*]

Sheet intent:
    (TP - mean(TP, 30)) / (std(TP, 30) + eps)
    where TP = (H + L + C) / 3

Implementation per protocol:
- Use Class A robust z-score (past-only) on TP directly with norm_window=30.
  (Robust z replaces mean/std z; median adapts similarly.)

Raw definition used:
    TP = (high + low + close)/3

Normalization policy:
- Robust z-score (past-only) window=30
- Gaussian clip + scale to [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from feature_utils import add_feature_per_symbol, class_a_signed


FEATURE_COL = "z_typical_price_30_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame, norm_window: int) -> pd.Series:
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    return class_a_signed(tp, norm_window)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: z_typical_price_30 (Class A signed, robust*)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-window", type=int, default=30)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(df, FEATURE_COL, lambda g: compute_feature(g, args.norm_window))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()