"""
Feature script: upper_wick_to_tr

Class: B (bounded semantic ratio)
Raw/semantic:
    (high - max(open,close)) / (TR + eps) -> [0,1]

NO z-score, NO robust z, NO Gaussian clip.

Output column:
- upper_wick_to_tr_pos_0_1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

EPS = 1e-12


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def upper_wick_to_tr_pos_0_1(df: pd.DataFrame) -> pd.Series:
    tr = true_range(df)
    upper = df["high"] - np.maximum(df["open"], df["close"])
    x = upper / (tr + EPS)
    return x.clip(0.0, 1.0)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    grouped = out.groupby("symbol", group_keys=False, sort=False) if "symbol" in out.columns else [(None, out)]
    parts = []

    for _, g in grouped:
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        final = upper_wick_to_tr_pos_0_1(g)
        g_out = g.copy()
        g_out["upper_wick_to_tr_pos_0_1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute upper_wick_to_tr feature (Class B) bounded in [0,1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}\nColumns added:\n - upper_wick_to_tr_pos_0_1")


if __name__ == "__main__":
    main()