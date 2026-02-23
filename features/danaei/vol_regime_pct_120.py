"""
Feature script: vol_regime_pct_120

Class: B (Bounded semantic — percentile rank of volatility)
Final scale: [0, 1]
z-score: No
Gaussian 60% clip: No

Purpose:
- Identifies the current volatility regime by ranking the recent 20-bar return-volatility
  against its own 120-bar history.
- 0 = lowest relative volatility regime (quiet market), 1 = highest (turbulent market).
- Percentile ranking is entirely scale-free and distribution-agnostic.

Raw formula (causal, past-only):
    ret1_t         = log(close_t / close_{t-1})   (1-bar log return)
    vol_20_t       = std(ret1_{t-20}..ret1_{t-1}) (past-only rolling 20-bar vol, ddof=1)
    vol_regime_pct_120_t =
        count(vol_20_{t-120}..vol_20_{t-1} <= vol_20_t) / 120

Anti-leakage:
- vol_20 rolling uses shift(1) so each value is based entirely on past candles.
- Percentile is computed against the PREVIOUS 120 values of vol_20.

Notes:
- Naturally bounded in [0, 1]; no z-score or Gaussian clipping (Class B).
- Computation is per-symbol; no cross-symbol mixing.
- Early rows NaN until sufficient history for both rolling windows is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def rolling_vol_20_past_only(df: pd.DataFrame, vol_window: int = 20) -> pd.Series:
    """
    Past-only 20-bar rolling volatility of 1-bar log returns.

    vol_20_t = std(ret1_{t-20}..ret1_{t-1}, ddof=1)
    """
    log_c = np.log(df["close"] + EPS)
    ret1 = log_c.diff(1)
    past_ret1 = ret1.shift(1)
    return past_ret1.rolling(window=vol_window, min_periods=vol_window).std(ddof=1)


def percentile_rank_past_only(series: pd.Series, window: int = 120) -> pd.Series:
    """
    Past-only percentile rank of `series` within the prior `window` values of itself.

    rank_t = count(series_{t-window}..series_{t-1} <= series_t) / window
    """
    vals = series.values
    n = len(vals)
    result = np.full(n, np.nan)

    for i in range(window, n):
        current = vals[i]
        if np.isnan(current):
            continue
        past = vals[i - window: i]
        valid_past = past[~np.isnan(past)]
        if len(valid_past) == 0:
            continue
        result[i] = np.sum(valid_past <= current) / window

    return pd.Series(result, index=series.index)


def build_feature_table(
    df: pd.DataFrame, vol_window: int = 20, pct_window: int = 120
) -> pd.DataFrame:
    """
    Append final feature column to the original dataset (no overwriting core OHLCV).

    Output column:
    - vol_regime_pct_120  (bounded semantic in [0, 1])
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

        vol_20 = rolling_vol_20_past_only(g, vol_window=vol_window)
        final = percentile_rank_past_only(vol_20, window=pct_window)

        g_out = g.copy()
        g_out["vol_regime_pct_120"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute vol_regime_pct_120 (Class B) — percentile of 20-bar vol over 120 bars in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input CSV with OHLCV columns.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--vol-window", type=int, default=20,
                   help="Rolling vol window (default: 20).")
    p.add_argument("--pct-window", type=int, default=120,
                   help="Percentile lookback window (default: 120).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(
        df, vol_window=args.vol_window, pct_window=args.pct_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns added:\n - vol_regime_pct_120")


if __name__ == "__main__":
    main()
