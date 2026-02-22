"""
Feature script: channel_pos_diff_20_100

Class: B (Bounded semantic oscillator / position-difference)
Final scale: [-1, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Compare "where price sits inside its recent range" over a short window vs a long window.
- Captures local overextension vs macro context:
    - positive: price is high in short range relative to its position in long range
    - negative: price is low in short range relative to its position in long range

Raw intuition + formula (causal, past-only):
1) Define past-only rolling channel for each window W:
      high_W_t = max(high_{t-W}..high_{t-1})
      low_W_t  = min(low_{t-W}..low_{t-1})

2) Define channel position (bounded in [0, 1]):
      pos_W_t = (close_t - low_W_t) / (high_W_t - low_W_t + eps)

3) Feature is the difference of short and long positions (bounded in [-1, 1]):
      channel_pos_diff_20_100_t = pos_20_t - pos_100_t

Anti-leakage:
- Rolling highs/lows are computed from past values only via shift(1) before rolling.

Notes:
- This is naturally bounded in [-1, 1] by construction; we apply a strict semantic
  clip to [-1, 1] (NOT Gaussian clipping) to enforce the Class B bounded domain.
- Early rows will be NaN due to rolling warm-up for the 100-bar channel.
- Computation is per-symbol and symbol-local; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def channel_position_past_only(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Past-only channel position in [0, 1]:
        pos_t = (close_t - low_W_t) / (high_W_t - low_W_t + eps)

    where:
        high_W_t = max(high_{t-W}..high_{t-1})
        low_W_t  = min(low_{t-W}..low_{t-1})
    """
    past_high = df["high"].shift(1)
    past_low = df["low"].shift(1)

    high_w = past_high.rolling(window=window, min_periods=window).max()
    low_w = past_low.rolling(window=window, min_periods=window).min()

    eps = 1e-12
    pos = (df["close"] - low_w) / ((high_w - low_w) + eps)

    # Enforce semantic bounds (range issues can occur with missing data or extreme gaps).
    return pos.clip(lower=0.0, upper=1.0)


def channel_pos_diff_20_100(df: pd.DataFrame, short_window: int = 20, long_window: int = 100) -> pd.Series:
    """
    Difference of short vs long past-only channel positions:
        pos_short - pos_long
    """
    pos_s = channel_position_past_only(df, window=short_window)
    pos_l = channel_position_past_only(df, window=long_window)
    return pos_s - pos_l


def semantic_clip_minus1_plus1(x: pd.Series) -> pd.Series:
    """
    Class B: enforce bounded semantic domain without z-scoring or Gaussian clipping.
    """
    return x.clip(lower=-1.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, short_window: int = 20, long_window: int = 100) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - channel_pos_diff_20_100 (bounded semantic in [-1, 1])
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

        raw = channel_pos_diff_20_100(g, short_window=short_window, long_window=long_window)
        final = semantic_clip_minus1_plus1(raw)

        g_out = g.copy()
        g_out["channel_pos_diff_20_100"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute channel_pos_diff_20_100 (Class B) as a leakage-safe bounded semantic feature in [-1, 1]."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with OHLCV columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--short-window", type=int, default=20, help="Short channel window (default: 20).")
    parser.add_argument("--long-window", type=int, default=100, help="Long channel window (default: 100).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    out = build_feature_table(
        df=df,
        short_window=args.short_window,
        long_window=args.long_window,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print("Columns added:")
    print(" - channel_pos_diff_20_100")


if __name__ == "__main__":
    main()