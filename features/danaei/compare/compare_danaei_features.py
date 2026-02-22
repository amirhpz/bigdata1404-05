"""
Build dataset with all 12 danaei features from ADAUSDT.csv and compare
against the reference ADAUSDT_with_features.csv to verify correctness.
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path('/home/mohammad/bigdata1404-05')
FEATURES_DIR = WORKSPACE / "features" / "danaei"
INPUT_CSV = WORKSPACE / "dataset" / "ADAUSDT.csv"
REF_CSV = WORKSPACE / "dataset" / "ADAUSDT_with_features.csv"
OUTPUT_CSV = WORKSPACE / "dataset" / "ADAUSDT_danaei_computed.csv"

# ── Helper: import a feature module by path ──────────────────────────────────


def import_feature(name: str):
    path = FEATURES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Column name mapping: computed → reference ────────────────────────────────
# Some feature scripts produce a bare name; the reference CSV uses a suffixed name.
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

# ── 1. Load raw OHLCV ────────────────────────────────────────────────────────
print(f"Loading {INPUT_CSV} …")
df = pd.read_csv(INPUT_CSV)
df = df[~df['datetime_utc'].astype(str).str.startswith('2025')].copy()
print(f"  Rows: {len(df):,}   Columns: {list(df.columns)}")

# ── 2. Apply all 12 features ─────────────────────────────────────────────────
print("\nComputing features …")
result = df.copy()
for mod_name in FEATURE_MODULES:
    mod = import_feature(mod_name)
    result = mod.build_feature_table(result)
    # find the new column(s) added
    new_cols = [c for c in result.columns if c not in df.columns
                and c not in [m for m in FEATURE_MODULES]]
    # just report progress
    added = [c for c in result.columns if c not in df.columns]
    print(
        f"  ✓ {mod_name:35s} → {[c for c in added if mod_name.split('_')[0] in c or any(w in c for w in mod_name.split('_'))]}")

# identify all computed feature columns (anything beyond original OHLCV)
computed_cols = [c for c in result.columns if c not in df.columns]
print(f"\n  Total feature columns computed: {len(computed_cols)}")
print(f"  Columns: {computed_cols}")

# ── 3. Save computed dataset ─────────────────────────────────────────────────
result.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved computed dataset → {OUTPUT_CSV}")

# ── 4. Load reference dataset ────────────────────────────────────────────────
print(f"\nLoading reference {REF_CSV} …")
ref = pd.read_csv(REF_CSV)
print(
    f"  Rows: {len(ref):,}   Feature columns present: {len(ref.columns) - len(df.columns)}")

# ── 5. Compare each feature column ──────────────────────────────────────────
print("\n" + "="*80)
print("FEATURE COMPARISON REPORT")
print("="*80)

ATOL = 1e-6   # absolute tolerance for "match"

summary_rows = []
all_ok = True

for computed_col, ref_col in COL_MAP.items():
    if computed_col not in result.columns:
        print(f"  [MISSING_COMPUTED]  {computed_col}")
        all_ok = False
        continue
    if ref_col not in ref.columns:
        print(
            f"  [MISSING_REFERENCE] {ref_col}  ← reference file has no such column")
        summary_rows.append({"feature": computed_col, "ref_col": ref_col,
                             "status": "REF_MISSING", "n_valid": 0,
                             "n_match": 0, "n_diff": 0,
                             "max_abs_err": float("nan"),
                             "mean_abs_err": float("nan")})
        continue

    s_comp = result[computed_col].reset_index(drop=True)
    s_ref = ref[ref_col].reset_index(drop=True)

    # align lengths
    min_len = min(len(s_comp), len(s_ref))
    s_comp = s_comp.iloc[:min_len]
    s_ref = s_ref.iloc[:min_len]

    both_valid = s_comp.notna() & s_ref.notna()
    n_valid = both_valid.sum()

    if n_valid == 0:
        print(
            f"  [NO_OVERLAP]  {computed_col:45s}  (no rows where both are non-NaN)")
        summary_rows.append({"feature": computed_col, "ref_col": ref_col,
                             "status": "NO_OVERLAP", "n_valid": 0,
                             "n_match": 0, "n_diff": 0,
                             "max_abs_err": float("nan"),
                             "mean_abs_err": float("nan")})
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
    print(f"  [{icon}] {computed_col:45s}  "
          f"valid={n_valid:6,}  match={pct_match:6.2f}%  "
          f"max_err={max_abs_err:.2e}  mean_err={mean_abs_err:.2e}")

    if status == "MISMATCH":
        # show first few differing positions
        diff_idx = np.where(abs_err > ATOL)[0]
        print(f"      First differing rows (up to 3):")
        for di in diff_idx[:3]:
            row_i = both_valid[both_valid].index[di]
            print(f"        row={row_i}  computed={v_comp[di]:.8f}  ref={v_ref[di]:.8f}  "
                  f"err={abs_err[di]:.2e}")

    summary_rows.append({
        "feature": computed_col, "ref_col": ref_col,
        "status": status, "n_valid": n_valid,
        "n_match": n_match, "n_diff": n_diff,
        "max_abs_err": max_abs_err, "mean_abs_err": mean_abs_err,
    })

print("="*80)
if all_ok:
    print("ALL FEATURES MATCH ✓")
else:
    print("SOME FEATURES DIFFER — see details above.")

# ── 6. Summary table ─────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
print("\nSummary table:")
print(summary_df.to_string(index=False))
