"""
Generate a new dataset with 12 danaei features from dataset-v1/ADAUSDT.csv,
save it to features/danaei/ADAUSDT_danaei_computed.csv,
and compare those features against the reference columns in dataset-v2/ADAUSDT.csv.

Usage:
    python generate_and_compare_danaei.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
FEATURES_DIR = WORKSPACE / "features" / "danaei"
INPUT_CSV = WORKSPACE / "dataset-v1" / "ADAUSDT.csv"
OUTPUT_CSV = FEATURES_DIR / "ADAUSDT_danaei_computed.csv"
REF_CSV = WORKSPACE / "dataset-v2" / "ADAUSDT.csv"

# ── Feature modules (12 danaei features) ─────────────────────────────────────
FEATURE_MODULES = [
    "bearish_engulf_score",
    "bullish_engulf_score",
    "channel_pos_20",
    "dollar_vol_rel_20",
    "efficiency_ratio_20",
    "log_range_over_vol_100",
    "percent_b_20",
    "range_compression_20_100",
    "vol_over_ema20",
    "vol_over_median20",
    "vol_regime_pct_120",
    "volume_percentile_60",
]

# ── Mapping: computed column name → reference column name in dataset-v2 ────────
# (the reference CSV uses slightly different suffixes)
COL_MAP = {
    "bearish_engulf_score_scaled_pos_0_1": "bearish_engulf_score_scaled_pos_0_1",
    "bullish_engulf_score_scaled_pos_0_1": "bullish_engulf_score_scaled_pos_0_1",
    "channel_pos_20": "channel_pos_20_semantic_pos_0_1",
    "dollar_vol_rel_20_scaled_pos_0_1": "dollar_vol_rel_20_scaled_pos_0_1",
    "efficiency_ratio_20": "efficiency_ratio_20_semantic_pos_0_1",
    "log_range_over_vol_100_scaled_pos_0_1": "log_range_over_vol_100_scaled_pos_0_1",
    "percent_b_20": "percent_b_20_semantic_signed_m1_p1",
    "range_compression_20_100": "range_compression_20_100_semantic_pos",
    "vol_over_ema20_scaled_pos_0_1": "vol_over_ema20_scaled_pos_0_1",
    "vol_over_median20_scaled_pos_0_1": "vol_over_median20_scaled_pos_0_1",
    "vol_regime_pct_120": "vol_regime_pct_120_semantic_pos_0_1",
    "volume_percentile_60": "volume_percentile_60_semantic_pos_0_1",
}


# ── Helper: import a feature module by path ───────────────────────────────────
def _import_feature(name: str):
    """Dynamically import a feature module from FEATURES_DIR."""
    path = FEATURES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # make sure the feature_utils sibling can be found
    sys.path.insert(0, str(FEATURES_DIR))
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load raw OHLCV from dataset-v1
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1 – Loading raw OHLCV from dataset-v1")
print("=" * 70)
print(f"  Input : {INPUT_CSV}")

if not INPUT_CSV.exists():
    sys.exit(f"ERROR: Input file not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

# Skip 2025 rows (keep only data up to end of 2024)
mask_2025 = df["datetime_utc"].astype(str).str.startswith("2025")
n_before = len(df)
df = df[~mask_2025].copy()
print(f"  Rows loaded   : {n_before:,}")
print(
    f"  Rows after    : {len(df):,}  (skipped {mask_2025.sum():,} rows from 2025)")
print(f"  Columns       : {list(df.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Apply all 12 danaei features
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 2 – Computing 12 danaei features")
print("=" * 70)

# remember original columns before feature addition
OHLCV_COLS = list(df.columns)
result = df.copy()

for mod_name in FEATURE_MODULES:
    mod = _import_feature(mod_name)
    result = mod.build_feature_table(result)
    new = [c for c in result.columns if c not in OHLCV_COLS]
    print(f"  ✓ {mod_name:<35s} → {new}")

computed_cols = [c for c in result.columns if c not in OHLCV_COLS]
print()
print(f"  Total feature columns computed : {len(computed_cols)}")
print(f"  Feature columns                : {computed_cols}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Save computed dataset to features/danaei/
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 3 – Saving computed dataset")
print("=" * 70)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved → {OUTPUT_CSV}")
print(f"  Shape  : {result.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Load reference dataset from dataset-v2
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 4 – Loading reference dataset from dataset-v2")
print("=" * 70)
print(f"  Reference : {REF_CSV}")

if not REF_CSV.exists():
    sys.exit(f"ERROR: Reference file not found: {REF_CSV}")

ref = pd.read_csv(REF_CSV)

# Filter reference to same date range (no 2025 rows)
ref_mask_2025 = ref["datetime_utc"].astype(str).str.startswith("2025")
ref = ref[~ref_mask_2025].copy()

print(f"  Rows (after removing 2025) : {len(ref):,}")
print(f"  Total columns              : {len(ref.columns)}")
ref_feature_cols = [c for c in ref.columns if c not in [
    "datetime_utc", "symbol", "open", "high", "low", "close", "volume"]]
print(f"  Feature columns in ref     : {len(ref_feature_cols)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Compare each of the 12 feature columns
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 5 – FEATURE COMPARISON REPORT")
print("=" * 70)

ATOL = 1e-6    # absolute tolerance for a "match"

summary_rows = []
all_ok = True

for computed_col, ref_col in COL_MAP.items():
    # ── check column presence ──
    if computed_col not in result.columns:
        print(f"  [MISSING_COMPUTED]  {computed_col}")
        all_ok = False
        summary_rows.append(dict(feature=computed_col, ref_col=ref_col,
                                 status="MISSING_COMPUTED",
                                 n_valid=0, n_match=0, n_diff=0,
                                 max_abs_err=float("nan"), mean_abs_err=float("nan")))
        continue

    if ref_col not in ref.columns:
        print(f"  [MISSING_REFERENCE] {ref_col}  ← not found in dataset-v2")
        summary_rows.append(dict(feature=computed_col, ref_col=ref_col,
                                 status="REF_MISSING",
                                 n_valid=0, n_match=0, n_diff=0,
                                 max_abs_err=float("nan"), mean_abs_err=float("nan")))
        continue

    # ── align on datetime_utc (inner join) for a fair comparison ──
    s_comp = (
        result[["datetime_utc", computed_col]]
        .set_index("datetime_utc")[computed_col]
    )
    s_ref = (
        ref[["datetime_utc", ref_col]]
        .set_index("datetime_utc")[ref_col]
    )
    common_idx = s_comp.index.intersection(s_ref.index)
    s_comp = s_comp.loc[common_idx]
    s_ref = s_ref.loc[common_idx]

    both_valid = s_comp.notna() & s_ref.notna()
    n_valid = int(both_valid.sum())

    if n_valid == 0:
        print(
            f"  [NO_OVERLAP]  {computed_col:<45s}  (no rows where both non-NaN)")
        summary_rows.append(dict(feature=computed_col, ref_col=ref_col,
                                 status="NO_OVERLAP",
                                 n_valid=0, n_match=0, n_diff=0,
                                 max_abs_err=float("nan"), mean_abs_err=float("nan")))
        continue

    v_comp = s_comp[both_valid].values.astype(float)
    v_ref = s_ref[both_valid].values.astype(float)

    abs_err = np.abs(v_comp - v_ref)
    n_match = int(np.sum(abs_err <= ATOL))
    n_diff = n_valid - n_match
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))

    status = "OK" if n_diff == 0 else "MISMATCH"
    if status != "OK":
        all_ok = False

    pct_match = 100.0 * n_match / n_valid
    icon = "✓" if status == "OK" else "✗"
    print(f"  [{icon}] {computed_col:<45s}  "
          f"valid={n_valid:6,}  match={pct_match:6.2f}%  "
          f"max_err={max_abs_err:.2e}  mean_err={mean_abs_err:.2e}")

    if status == "MISMATCH":
        diff_idx = np.where(abs_err > ATOL)[0]
        print(f"      First differing rows (up to 3):")
        for di in diff_idx[:3]:
            row_ts = common_idx[both_valid][di]
            print(f"        ts={row_ts}  "
                  f"computed={v_comp[di]:.8f}  ref={v_ref[di]:.8f}  "
                  f"err={abs_err[di]:.2e}")

    summary_rows.append(dict(
        feature=computed_col, ref_col=ref_col,
        status=status, n_valid=n_valid,
        n_match=n_match, n_diff=n_diff,
        max_abs_err=max_abs_err, mean_abs_err=mean_abs_err,
    ))

print("=" * 70)
if all_ok:
    print("ALL FEATURES MATCH ✓")
else:
    print("SOME FEATURES DIFFER — see details above.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Summary table
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 6 – Summary table")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
