"""
Feature script: run_balance_5

Class: B (Bounded semantic oscillator / pattern-like rule)
Final scale: [-1, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Up vs down run balance over the past 5 bars:
      (count(up,5) - count(down,5)) / 5
- Interpretable as bearish (-1) / neutral (0) / bullish (+1) run tendency.

Definitions (causal):
- ret1_t = (close_t / close_{t-1}) - 1
- up_t   = 1 if ret1_t > 0 else 0
- down_t = 1 if ret1_t < 0 else 0
- run_balance_5_t = (sum(up_{t-5}..up_{t-1}) - sum(down_{t-5}..down_{t-1})) / 5

Anti-leakage:
- The rolling counts are computed from past values only:
  we shift the up/down indicator by 1 before rolling.
  This prevents the current candle from contributing to its own feature value.

Notes:
- This is naturally bounded in [-1, 1] by construction; we apply a strict semantic
  clip to [-1, 1] (NOT Gaussian clipping) to enforce the Class B bounded domain.
- Early rows will be NaN due to return warm-up and rolling warm-up.
- Computation is per-symbol and symbol-local; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ret_1(df: pd.DataFrame) -> pd.Series:
    """
    One-bar simple return:
        ret1_t = close_t / close_{t-1} - 1
    """
    prev_close = df["close"].shift(1)
    eps = 1e-12
    return (df["close"] / (prev_close + eps)) - 1.0


def run_balance_5(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Leakage-safe run balance:
        (count(up, lookback) - count(down, lookback)) / lookback

    where counts are computed over the past-only window:
        t-lookback .. t-1
    """
    r1 = ret_1(df)

    up = (r1 > 0).astype("float64")
    down = (r1 < 0).astype("float64")

    # Past-only counts to avoid leakage:
    up_past = up.shift(1).rolling(window=lookback, min_periods=lookback).sum()
    down_past = down.shift(1).rolling(window=lookback, min_periods=lookback).sum()

    return (up_past - down_past) / float(lookback)


def semantic_clip_minus1_plus1(x: pd.Series) -> pd.Series:
    """
    Class B: enforce bounded semantic domain without z-scoring or Gaussian clipping.
    """
    return x.clip(lower=-1.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - run_balance_5 (bounded semantic in [-1, 1])
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    # Compute per-symbol to prevent cross-symbol contamination.
    if "symbol" in out.columns:
        grouped = out.groupby("symbol", group_keys=False, sort=False)
    else:
        grouped = [(None, out)]

    parts = []
    for _, g in grouped:
        # Ensure causal order inside each symbol timeline.
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        raw = run_balance_5(g, lookback=lookback)
        final = semantic_clip_minus1_plus1(raw)

        g_out = g.copy()
        g_out["run_balance_5"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute run_balance_5 (Class B) as a leakage-safe bounded semantic feature in [-1, 1]."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with OHLCV columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--lookback", type=int, default=5, help="Rolling lookback (default: 5).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    out = build_feature_table(df=df, lookback=args.lookback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print("Columns added:")
    print(" - run_balance_5")


if __name__ == "__main__":
    main()