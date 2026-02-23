"""
Feature: channel_pos_diff_20_100
Class: B (bounded semantic, signed)

Raw definition:
    pos_20 - pos_100

Where (past-only channels):
    high_W_t = max(high_{t-W}..high_{t-1})
    low_W_t  = min(low_{t-W}..low_{t-1})
    pos_W_t  = (close_t - low_W_t) / (high_W_t - low_W_t + eps)   in [0, 1]

Processing policy (Class B):
- No z-score / robust z-score
- No Gaussian clipping
- Semantic bounded result in [-1, +1] (by construction); apply a strict safety clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol


FEATURE_COL = "channel_pos_diff_20_100"


def _pos_past_only(g: pd.DataFrame, window: int) -> pd.Series:
    """Past-only channel position in [0, 1]."""
    high_w = g["high"].shift(1).rolling(window=window, min_periods=window).max()
    low_w = g["low"].shift(1).rolling(window=window, min_periods=window).min()
    pos = (g["close"] - low_w) / ((high_w - low_w) + EPS)
    return pos.clip(0.0, 1.0)


def compute_feature(g: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    """Compute feature values for one symbol slice and return final bounded column."""
    pos_s = _pos_past_only(g, short_window)
    pos_l = _pos_past_only(g, long_window)
    raw = pos_s - pos_l  # naturally in [-1, 1]
    return raw.clip(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature: channel_pos_diff_20_100 (Class B signed)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--short-window", type=int, default=20)
    p.add_argument("--long-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = add_feature_per_symbol(
        df,
        FEATURE_COL,
        lambda g: compute_feature(g, args.short_window, args.long_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()