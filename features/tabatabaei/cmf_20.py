"""
Feature script: cmf_20

Class: B (Bounded semantic oscillator / money-flow)
Final scale: [-1, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Close-in-range weighted money flow, normalized by total volume.
- Interpretable as selling (-1) / balanced (0) / buying (+1) pressure.

Raw intuition + formula (causal, past-only):
- Money Flow Multiplier (MFM):
      MFM_t = ((close_t - low_t) - (high_t - close_t)) / (high_t - low_t + eps)
            = (2*close_t - high_t - low_t) / (high_t - low_t + eps)

- Money Flow Volume (MFV):
      MFV_t = MFM_t * volume_t

- Chaikin Money Flow (CMF) over lookback L=20 (PAST-ONLY to prevent leakage):
      cmf_20_t = sum(MFV_{t-L}..MFV_{t-1}) / (sum(volume_{t-L}..volume_{t-1}) + eps)

Notes:
- CMF is naturally bounded approximately in [-1, 1]. We apply a strict semantic clip
  to [-1, 1] (NOT Gaussian clipping) to enforce the Class B bounded domain.
- Early rows will be NaN due to rolling warm-up (min_periods=L). This is expected.
- Computation is per-symbol and symbol-local; no cross-symbol normalization/mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def money_flow_multiplier(df: pd.DataFrame) -> pd.Series:
    """
    Money Flow Multiplier (MFM), bounded in approximately [-1, 1].

    MFM_t = ((close_t - low_t) - (high_t - close_t)) / (high_t - low_t + eps)
          = (2*close_t - high_t - low_t) / (high_t - low_t + eps)
    """
    eps = 1e-12
    hl_range = (df["high"] - df["low"]).abs()
    return (2.0 * df["close"] - df["high"] - df["low"]) / (hl_range + eps)


def cmf_20(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Leakage-safe Chaikin Money Flow (CMF) computed using PAST-ONLY rolling sums.

    cmf_t = sum(MFM * V, lookback) / (sum(V, lookback) + eps)

    Anti-leakage:
    - Rolling sums use .shift(1) so the current candle is never included.
    """
    mfm = money_flow_multiplier(df)
    mfv = mfm * df["volume"]

    past_mfv = mfv.shift(1)
    past_vol = df["volume"].shift(1)

    sum_mfv = past_mfv.rolling(window=lookback, min_periods=lookback).sum()
    sum_vol = past_vol.rolling(window=lookback, min_periods=lookback).sum()

    eps = 1e-12
    return sum_mfv / (sum_vol + eps)


def semantic_clip_minus1_plus1(x: pd.Series) -> pd.Series:
    """
    Class B: enforce bounded semantic domain without z-scoring or Gaussian clipping.
    """
    return x.clip(lower=-1.0, upper=1.0)


def build_feature_table(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - cmf_20   (bounded semantic in [-1, 1])
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

        raw = cmf_20(g, lookback=lookback)
        final = semantic_clip_minus1_plus1(raw)

        g_out = g.copy()
        g_out["cmf_20"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cmf_20 (Class B) as a leakage-safe bounded semantic feature in [-1, 1]."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with OHLCV columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--lookback", type=int, default=20, help="Rolling lookback (default: 20).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    out = build_feature_table(df=df, lookback=args.lookback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print("Columns added:")
    print(" - cmf_20")


if __name__ == "__main__":
    main()