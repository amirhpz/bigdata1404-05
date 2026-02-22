"""
Feature script: ema_gap_atr_20

Purpose:
- Provide a reference implementation that teammates can copy for other features.

Pipeline in this file:
1) Computes a relative feature:
      ema_gap_atr_20 = (close - EMA(close, 20)) / ATR(14)
2) Applies leakage-free rolling robust normalization (median/MAD from past only).
3) Clips normalized values to the central 60% Gaussian interval [-0.8416, 0.8416].
4) Converts to final fuzzy-ready scale in [-1, 1] (because this feature is signed).

Scaling rule for all future features:
- Signed feature (can be negative/positive) -> scale to [-1, 1]
- Positive-domain feature (always >= 0) -> scale to [0, 1]

Note on early rows:
- Initial rows can be NaN due to EMA/ATR warm-up and rolling normalization window.
- This is expected and should be handled later in the training/inference pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Central 60% clipping limits of standard Gaussian:
# P(Z <= 0.8416) ~= 0.80 and P(Z <= -0.8416) ~= 0.20
# So values are clipped to the middle 60% mass.
GAUSSIAN_CLIP_LOW = -0.8416
GAUSSIAN_CLIP_HIGH = 0.8416


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute True Range (TR) for each candle.

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
    Compute ATR using EMA smoothing.

    We use min_periods=period so ATR values before enough history remain NaN.
    This is intentional to avoid unstable early estimates.
    """
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def ema_gap_atr_20(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14) -> pd.Series:
    """
    Relative signed feature:
        (close - EMA(close, ema_period)) / ATR(atr_period)

    Interpretation:
    - > 0: price is above EMA by some ATR units
    - < 0: price is below EMA by some ATR units
    """
    ema = df["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    atr_series = atr(df, period=atr_period)

    # Small epsilon to avoid division by zero in edge cases.
    eps = 1e-12
    return (df["close"] - ema) / (atr_series + eps)


def rolling_robust_zscore_past_only(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Leakage-free rolling robust z-score.

    Important anti-leakage rule:
    - Median/MAD are computed from past values only, by shifting 1 step.
    - Current value is NOT included in its own normalization stats.
    - This keeps normalization leakage-free for backtests and model training.
    """
    past = series.shift(1)
    median = past.rolling(window=window, min_periods=window).median()

    # MAD_t = median(|x - median(x)|) over the same past-only rolling window
    abs_dev = (past - median).abs()
    mad = abs_dev.rolling(window=window, min_periods=window).median()

    # 0.67448975 makes robust z approximately comparable to standard z for Gaussian data.
    gaussian_consistency = 0.67448975

    eps = 1e-12
    return gaussian_consistency * (series - median) / (mad + eps)


def clip_central_gaussian_60(z: pd.Series) -> pd.Series:
    """
    Clip normalized values to central Gaussian 60% interval.
    """
    return z.clip(lower=GAUSSIAN_CLIP_LOW, upper=GAUSSIAN_CLIP_HIGH)


def clipped_signed_to_minus1_plus1(clipped_z: pd.Series) -> pd.Series:
    """
    Convert clipped z-score to [-1, +1] for a signed feature.

    Since clipped_z is bounded in [-0.8416, +0.8416], dividing by 0.8416 maps it
    directly to [-1, +1].
    """
    return clipped_z / GAUSSIAN_CLIP_HIGH


def build_feature_table(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    norm_window: int = 100,
) -> pd.DataFrame:
    """
    Build output table by keeping the original dataset and appending
    the final feature as the last column.

    Output column:
    - ema_gap_atr_20_scaled_signed_m1_p1 (range approximately [-1, 1] after clipping)
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    # Avoid cross-symbol leakage:
    # compute feature + normalization independently inside each symbol history.
    if "symbol" in out.columns:
        grouped = out.groupby("symbol", group_keys=False, sort=False)
    else:
        grouped = [(None, out)]

    result_parts = []
    for _, g in grouped:
        # Ensure causal order inside each symbol timeline.
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        raw_feature = ema_gap_atr_20(g, ema_period=ema_period, atr_period=atr_period)
        robust_z = rolling_robust_zscore_past_only(raw_feature, window=norm_window)
        z_clipped = clip_central_gaussian_60(robust_z)
        final_scaled = clipped_signed_to_minus1_plus1(z_clipped)

        g_out = g.copy()
        g_out["ema_gap_atr_20_scaled_signed_m1_p1"] = final_scaled
        result_parts.append(g_out)

    out = pd.concat(result_parts).sort_index()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ema_gap_atr_20 feature and map it to signed [-1, +1]."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with OHLCV columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--ema-period", type=int, default=20, help="EMA period (default: 20).")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period (default: 14).")
    parser.add_argument(
        "--norm-window",
        type=int,
        default=100,
        help="Past-only rolling window for robust z-score normalization (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Build final output column (intermediate stages are intentionally not exported)
    out = build_feature_table(
        df=df,
        ema_period=args.ema_period,
        atr_period=args.atr_period,
        norm_window=args.norm_window,
    )

    # Ensure output folder exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Short terminal summary to confirm completion
    print(f"Saved: {args.output}")
    print("Columns added:")
    print(" - ema_gap_atr_20_scaled_signed_m1_p1")


if __name__ == "__main__":
    main()
