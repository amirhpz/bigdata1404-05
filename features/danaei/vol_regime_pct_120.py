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
from feature_utils import EPS, add_feature_per_symbol

FEATURE_COL = "vol_regime_pct_120"


def _rolling_vol_20(g: pd.DataFrame, vol_window: int = 20) -> pd.Series:
    """Past-only rolling `vol_window`-bar return volatility."""
    log_c = np.log(g["close"] + EPS)
    ret1 = log_c.diff(1)
    return ret1.shift(1).rolling(window=vol_window, min_periods=vol_window).std(ddof=1)


def _percentile_rank(series: pd.Series, window: int = 120) -> pd.Series:
    """Past-only percentile rank of series within prior `window` values."""
    vals = series.values
    n = len(vals)
    result = np.full(n, np.nan)
    for i in range(window, n):
        current = vals[i]
        if np.isnan(current):
            continue
        past = vals[i - window: i]
        valid = past[~np.isnan(past)]
        if len(valid) == 0:
            continue
        result[i] = np.sum(valid <= current) / window
    return pd.Series(result, index=series.index)


def compute_feature(g: pd.DataFrame, vol_window: int = 20, pct_window: int = 120) -> pd.Series:
    """Percentile of rolling vol over `pct_window` bars."""
    vol_20 = _rolling_vol_20(g, vol_window=vol_window)
    return _percentile_rank(vol_20, window=pct_window)


def build_feature_table(
    df: pd.DataFrame, vol_window: int = 20, pct_window: int = 120
) -> pd.DataFrame:
    return add_feature_per_symbol(
        df, FEATURE_COL, lambda g: compute_feature(g, vol_window, pct_window)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute vol_regime_pct_120 (Class B) — percentile of 20-bar vol over 120 bars in [0, 1]."
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--pct-window", type=int, default=120)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_feature_table(
        df, vol_window=args.vol_window, pct_window=args.pct_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Columns added:\n - {FEATURE_COL}")


if __name__ == "__main__":
    main()
