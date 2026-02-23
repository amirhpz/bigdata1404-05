"""
Feature script: bullish_engulf_score

Class: A (Open-domain, shock-sensitive, positive — sparse pattern signal)
Final scale: [0, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Quantifies the intensity of a bullish engulfing pattern.
- Zero unless the candle qualifies as a bullish engulf; positive when it does.
- ATR-scaling makes the score cross-symbol and time-scale invariant.
- Class A because the raw series is sparse (mostly zero) with an open-domain tail.

Raw formula (causal):
    A "bullish engulf" candle t requires (ALL must hold):
        1) prev candle is bearish:      close_{t-1} < open_{t-1}
        2) current candle is bullish:   close_t > open_t
        3) current open engulfs prev:   open_t  <= close_{t-1}  (opens at or below prev close)
        4) current close engulfs prev:  close_t >= open_{t-1}   (closes at or above prev open)

    bullish_engulf_score_raw_t =
        I[bull_engulf_t] * abs(close_t - open_t) / (ATR_14_t + eps)

    where ATR_14_t = EMA(TR, 14) — causal ATR.

Class A pipeline:
1) Compute raw score causally.
2) Rolling robust z-score (past-only median/MAD).
3) Clip to [-0.8416, +0.8416].
4) Scale to [0, 1] (positive-domain).

Notes:
- Computation is per-symbol; no cross-symbol mixing.
- Sparse feature: most rows are zero. Class A handles heavy tail from non-zero events.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feature_utils import EPS, add_feature_per_symbol, atr, class_a_positive

FEATURE_COL = "bullish_engulf_score_scaled_pos_0_1"


def compute_feature(g: pd.DataFrame, atr_period: int = 14, norm_window: int = 100) -> pd.Series:
    """Bullish engulf intensity normalized via Class A pipeline."""
    prev_open = g["open"].shift(1)
    prev_close = g["close"].shift(1)
    is_engulf = (
        (prev_close < prev_open)        # prev bearish
        & (g["close"] > g["open"])      # curr bullish
        & (g["open"] <= prev_close)     # open engulfs prev close
        & (g["close"] >= prev_open)     # close engulfs prev open
    )
    body = (g["close"] - g["open"]).abs()
    atr_14 = atr(g, period=atr_period)
    raw = is_engulf.astype(float) * body / (atr_14 + EPS)
    return class_a_positive(raw, norm_window)


def build_feature_table(
    df: pd.DataFrame, atr_period: int = 14, norm_window: int = 100
) -> pd.DataFrame:
    return add_feature_per_symbol(
        df, FEATURE_COL, lambda g: compute_feature(g, atr_period, norm_window)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute bullish_engulf_score (Class A) scaled to [0, 1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--norm-window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(
        df, atr_period=args.atr_period, norm_window=args.norm_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
