"""
Feature script: channel_pos_20

Class: B (Bounded semantic oscillator / channel position)
Final scale: [0, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Where is the current close located within the prior 20-bar high-low channel?
- 0 = at the bottom of the 20-bar range, 1 = at the top of the 20-bar range.
- Provides a scale-invariant, symbol-agnostic measure of near-term relative position.

Raw formula (causal, past-only):
    high_20_t = max(high_{t-20}..high_{t-1})   (PAST-ONLY via shift(1))
    low_20_t  = min(low_{t-20}..low_{t-1})
    channel_pos_20_t = (close_t - low_20_t) / (high_20_t - low_20_t + eps)

Notes:
- Naturally bounded in [0, 1] by construction; we apply a semantic clip to enforce it.
- No z-score or Gaussian clipping (Class B).
- Rolling is shift(1)-based to avoid leakage.
- Computation is per-symbol; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EPS = 1e-12


def channel_pos_20(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Past-only channel position in [0, 1].

    channel_pos_t = (close_t - low_w_t) / (high_w_t - low_w_t + eps)

    where:
        high_w_t = max(high_{t-window}..high_{t-1})
        low_w_t  = min(low_{t-window}..low_{t-1})
    """
    past_high = df["high"].shift(1)
    past_low = df["low"].shift(1)

    high_w = past_high.rolling(window=window, min_periods=window).max()
    low_w = past_low.rolling(window=window, min_periods=window).min()

    raw = (df["close"] - low_w) / ((high_w - low_w) + EPS)
    return raw.clip(lower=0.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - channel_pos_20  (bounded semantic in [0, 1])
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
        g_out["channel_pos_20"] = channel_pos_20(g, window=window)
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute channel_pos_20 (Class B) — past-only 20-bar channel position in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--window", type=int, default=20,
                   help="Channel lookback window (default: 20).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - channel_pos_20")


if __name__ == "__main__":
    main()
