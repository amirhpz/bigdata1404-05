"""
Feature: body_signed_to_tr
Class: B (bounded semantic ratio)

Raw definition:
    (close - open) / (TR + eps)

Normalization policy:
- No z-score (semantic bounded ratio)
- No Gaussian clipping
- Keep semantic bounded ratio and apply strict clip to [-1, +1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_utils import EPS, add_feature_per_symbol, true_range

FEATURE_COL = "body_signed_to_tr_scaled_signed_m1_p1"


def compute_feature(g: pd.DataFrame) -> pd.Series:
    tr = true_range(g)  # >= 0
    raw = (g["close"] - g["open"]) / (tr + EPS)
    # By construction, |close-open| <= TR, so raw should lie in [-1,1] (eps makes it slightly smaller).
    return raw.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute body_signed_to_tr (Class B).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df=df,
        feature_col=FEATURE_COL,
        compute_fn=lambda g: compute_feature(g),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:")
    print(f" - {FEATURE_COL}")


if __name__ == "__main__":
    main()