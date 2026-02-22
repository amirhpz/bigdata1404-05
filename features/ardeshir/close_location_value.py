"""
Feature script: close_location_value

Class: B (bounded semantic position)
Raw/semantic:
    (close - low) / (high - low + eps) -> [0,1]

NO z-score, NO robust z, NO Gaussian clip.

Output column:
- close_location_value_pos_0_1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

EPS = 1e-12


def close_location_value_pos_0_1(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"])
    x = (df["close"] - df["low"]) / (rng + EPS)
    return x.clip(0.0, 1.0)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    grouped = out.groupby("symbol", group_keys=False, sort=False) if "symbol" in out.columns else [(None, out)]
    parts = []

    for _, g in grouped:
        if "datetime_utc" in g.columns:
            g = g.sort_values("datetime_utc")

        final = close_location_value_pos_0_1(g)
        g_out = g.copy()
        g_out["close_location_value_pos_0_1"] = final
        parts.append(g_out)

    return pd.concat(parts).sort_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute close_location_value feature (Class B) bounded in [0,1].")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out = build_feature_table(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}\nColumns added:\n - close_location_value_pos_0_1")


if __name__ == "__main__":
    main()