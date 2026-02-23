"""
Feature script: efficiency_ratio_20

Class: B (Bounded semantic — Kaufman Efficiency Ratio)
Final scale: [0, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Measures how efficiently price has moved over the past 20 bars.
- ER = net price change / total path length travelled.
- 0 = perfectly choppy (all movement reversed), 1 = perfectly directional (straight line).
- Scale-invariant because both numerator and denominator are in price units (ratio).

Raw formula (causal, past-only):
    net_move_t   = abs(close_t - close_{t-20})             (net displacement)
    path_t       = sum(abs(close_{t-k} - close_{t-k-1}),
                        k=0..19)                           (sum of 1-bar abs moves over past 20 bars)
    efficiency_ratio_20_t = net_move_t / (path_t + eps)

Anti-leakage:
- close_{t-20} and close_{t-1} are both past values. No future data used.
- path_t uses shifts entirely within past 20 candles.

Notes:
- Naturally bounded in [0, 1]; semantic clip applied as a safeguard.
- No z-score or Gaussian clipping (Class B).
- Computation is per-symbol; no cross-symbol mixing.
- Early rows NaN until 21 candles of history are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EPS = 1e-12


def efficiency_ratio_20(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Kaufman Efficiency Ratio over past `window` bars.

    ER_t = abs(close_t - close_{t-window}) / (path_length_t + eps)

    path_length_t = sum of abs(close[i] - close[i-1]) for i in [t-window+1 .. t]
    """
    close = df["close"]

    # Net displacement: diff over window bars
    net_move = (close - close.shift(window)).abs()

    # Path length: sum of absolute daily changes over the window
    abs_daily = close.diff(1).abs()
    path_len = abs_daily.rolling(window=window, min_periods=window).sum()

    raw = net_move / (path_len + EPS)
    return raw.clip(lower=0.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - efficiency_ratio_20  (bounded semantic in [0, 1])
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    if "symbol" in out.columns:
        grouped = out.groupby("symbol", group_keys=False, sort=False)
    else:
        grouped = [(None, out)]

    parts = []
    for _, g in grouped:
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        g_out = g.copy()
        g_out["efficiency_ratio_20"] = efficiency_ratio_20(g, window=window)
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute efficiency_ratio_20 (Class B) — Kaufman ER in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--window", type=int, default=20,
                   help="Lookback window (default: 20).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - efficiency_ratio_20")


if __name__ == "__main__":
    main()
