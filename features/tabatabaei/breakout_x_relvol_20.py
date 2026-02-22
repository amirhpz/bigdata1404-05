"""
Feature script: breakout_x_relvol_20

Class: A (Open-domain, shock-sensitive)
Final scale: signed [-1, 1]
Normalization: past-only rolling robust z (median/MAD), per-symbol
Gaussian 60% clip: Yes ([-0.8416, +0.8416])

Purpose:
- Breakout "quality" combining ATR-scaled distance above/below prior 20-bar high
  and current relative volume.
- Interpretable as:
    - negative: below prior high (failed/weak breakout)
    - near 0: near prior high
    - positive: above prior high (breakout), amplified when volume is high vs its EMA

Raw feature (causal):
1) prev20High_t = max(high_{t-20}..high_{t-1})   (PAST-ONLY)
2) ATR14_t via EMA smoothing (causal)
3) rel_vol_t = volume_t / (EMA(volume,20)_t + eps)
4) raw_t = ((close_t - prev20High_t) / (ATR14_t + eps)) * rel_vol_t

Class A pipeline (mandatory):
- robust z-score using past-only rolling median/MAD (shift(1) before rolling)
- clip to [-0.8416, +0.8416]
- scale to [-1, +1]

Notes:
- Computation is per-symbol and symbol-local; no cross-symbol mixing.
- Early rows will be NaN due to rolling/EMA warm-up and normalization warm-up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Central 60% clipping limits of standard Gaussian:
GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range (TR) for each candle.

    TR_t = max(
        high_t - low_t,
        abs(high_t - close_{t-1}),
        abs(low_t  - close_{t-1})
    )
    """
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR via EMA smoothing of True Range.

    min_periods=period keeps early values NaN (intentional warm-up).
    """
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    EMA helper with min_periods=span to keep early values NaN (intentional warm-up).
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_prev_high_past_only(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Past-only rolling maximum of high:
        prev_high_t = max(high_{t-window}..high_{t-1})

    Anti-leakage:
    - Uses shift(1) so current candle is NOT included.
    """
    past_high = df["high"].shift(1)
    return past_high.rolling(window=window, min_periods=window).max()


def breakout_x_relvol_20(df: pd.DataFrame, high_window: int = 20, atr_period: int = 14, vol_ema_span: int = 20) -> pd.Series:
    """
    Relative signed feature:
        ((close - prev20High) / ATR14) * (volume / EMA(volume,20))
    """
    prev_high = rolling_prev_high_past_only(df, window=high_window)
    atr_series = atr(df, period=atr_period)

    eps = 1e-12
    rel_vol = df["volume"] / (ema(df["volume"], span=vol_ema_span) + eps)

    return ((df["close"] - prev_high) / (atr_series + eps)) * rel_vol


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
    high_window: int = 20,
    atr_period: int = 14,
    vol_ema_span: int = 20,
    norm_window: int = 100,
) -> pd.DataFrame:
    """
    Append final feature column to original dataset.

    Output column:
    - breakout_x_relvol_20_scaled_signed_m1_p1
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

        raw = breakout_x_relvol_20(
            g,
            high_window=high_window,
            atr_period=atr_period,
            vol_ema_span=vol_ema_span,
        )
        rz = rolling_robust_zscore_past_only(raw, window=norm_window)
        rz_clip = clip_central_gaussian_60(rz)
        final = clipped_signed_to_minus1_plus1(rz_clip)

        g_out = g.copy()
        g_out["breakout_x_relvol_20_scaled_signed_m1_p1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute breakout_x_relvol_20 (Class A) and map it to signed [-1, +1] with robust-z + Gaussian clip."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with OHLCV columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--high-window", type=int, default=20, help="Past rolling high window (default: 20).")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period (default: 14).")
    parser.add_argument("--vol-ema-span", type=int, default=20, help="EMA span for volume (default: 20).")
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
        high_window=args.high_window,
        atr_period=args.atr_period,
        vol_ema_span=args.vol_ema_span,
        norm_window=args.norm_window,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print("Columns added:")
    print(" - breakout_x_relvol_20_scaled_signed_m1_p1")


if __name__ == "__main__":
    main()