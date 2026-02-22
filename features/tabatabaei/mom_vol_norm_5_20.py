"""
Feature script: mom_vol_norm_5_20

Class: A (Open-domain, shock-sensitive)
Final scale: signed [-1, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Short momentum per medium risk:
      (close / close[-5] - 1) / (std(ret1, 20) + eps)
- Interpretable as weak/normal/strong short thrust relative to recent return volatility,
  then made fuzzy-ready via robust-z + Gaussian central clipping.

Raw feature (causal):
1) mom_5_t = close_t / close_{t-5} - 1     (uses only past close)
2) ret1_t  = close_t / close_{t-1} - 1
3) vol_20_t = std(ret1_{t-20}..ret1_{t-1})  (PAST-ONLY)
4) raw_t = mom_5_t / (vol_20_t + eps)

Important anti-leakage rule:
- The rolling std for volatility is computed from past returns only by shifting ret1 by 1
  before rolling.

Class A pipeline (mandatory):
- robust z-score using past-only rolling median/MAD (shift(1) before rolling)
- clip to [-0.8416, +0.8416]
- scale to [-1, +1]

Notes:
- Early rows will be NaN due to momentum/volatility warm-up and normalization warm-up.
- Computation is per-symbol and symbol-local; no cross-symbol mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Central 60% clipping limits of standard Gaussian:
GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416


def ret_1(df: pd.DataFrame) -> pd.Series:
    """
    One-bar simple return:
        ret1_t = close_t / close_{t-1} - 1
    """
    prev_close = df["close"].shift(1)
    eps = 1e-12
    return (df["close"] / (prev_close + eps)) - 1.0


def momentum_n(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """
    N-bar simple momentum:
        mom_n_t = close_t / close_{t-n} - 1
    """
    past_close = df["close"].shift(n)
    eps = 1e-12
    return (df["close"] / (past_close + eps)) - 1.0


def rolling_std_past_only(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Leakage-safe rolling standard deviation over a past-only window.

    Anti-leakage:
    - Uses series.shift(1) so current value is NOT included in the rolling window.
    """
    past = series.shift(1)
    return past.rolling(window=window, min_periods=window).std()


def mom_vol_norm_5_20(df: pd.DataFrame, mom_lookback: int = 5, vol_lookback: int = 20) -> pd.Series:
    """
    Relative signed feature:
        (close/close[-mom_lookback] - 1) / (std(ret1, vol_lookback) + eps)

    Volatility std is computed past-only to prevent leakage.
    """
    mom = momentum_n(df, n=mom_lookback)
    r1 = ret_1(df)
    vol = rolling_std_past_only(r1, window=vol_lookback)

    eps = 1e-12
    return mom / (vol + eps)


def rolling_robust_zscore_past_only(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Leakage-free rolling robust z-score:
      rz_t = 0.67448975 * (x_t - median_past) / (MAD_past + eps)

    Anti-leakage:
    - median/MAD computed from past values only (series.shift(1) before rolling).
    """
    past = series.shift(1)
    median = past.rolling(window=window, min_periods=window).median()

    abs_dev = (past - median).abs()
    mad = abs_dev.rolling(window=window, min_periods=window).median()

    gaussian_consistency = 0.67448975
    eps = 1e-12
    return gaussian_consistency * (series - median) / (mad + eps)


def clip_central_gaussian_60(z: pd.Series) -> pd.Series:
    """Clip normalized values to central Gaussian 60% interval."""
    return z.clip(lower=GAUSSIAN_CLIP_LOW, upper=GAUSSIAN_CLIP_HIGH)


def clipped_signed_to_minus1_plus1(clipped_z: pd.Series) -> pd.Series:
    """
    Convert clipped z-score to [-1, +1] for a signed Class A feature.
    """
    return clipped_z / GAUSSIAN_CLIP_HIGH


def build_feature_table(
    df: pd.DataFrame,
    mom_lookback: int = 5,
    vol_lookback: int = 20,
    norm_window: int = 100,
) -> pd.DataFrame:
    """
    Append final feature column to original dataset.

    Output column:
    - mom_vol_norm_5_20_scaled_signed_m1_p1
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    # Avoid cross-symbol leakage: compute independently per symbol.
    if "symbol" in out.columns:
        grouped = out.groupby("symbol", group_keys=False, sort=False)
    else:
        grouped = [(None, out)]

    parts = []
    for _, g in grouped:
        # Ensure causal order inside each symbol timeline.
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        raw = mom_vol_norm_5_20(g, mom_lookback=mom_lookback, vol_lookback=vol_lookback)
        rz = rolling_robust_zscore_past_only(raw, window=norm_window)
        rz_clip = clip_central_gaussian_60(rz)
        final = clipped_signed_to_minus1_plus1(rz_clip)

        g_out = g.copy()
        g_out["mom_vol_norm_5_20_scaled_signed_m1_p1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mom_vol_norm_5_20 (Class A) and map it to signed [-1, +1] with robust-z + Gaussian clip."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with OHLCV columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--mom-lookback", type=int, default=5, help="Momentum lookback (default: 5).")
    parser.add_argument("--vol-lookback", type=int, default=20, help="Volatility std lookback (default: 20).")
    parser.add_argument(
        "--norm-window",
        type=int,
        default=100,
        help="Past-only rolling window for robust z-score normalization (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    out = build_feature_table(
        df=df,
        mom_lookback=args.mom_lookback,
        vol_lookback=args.vol_lookback,
        norm_window=args.norm_window,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print("Columns added:")
    print(" - mom_vol_norm_5_20_scaled_signed_m1_p1")


if __name__ == "__main__":
    main()