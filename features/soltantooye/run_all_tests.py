"""
Automated Feature Testing Suite
Runs all feature scripts and tests them against reference file.

Usage:
    python run_all_tests.py
"""
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


# Define all features with their script names
FEATURES = [
    {
        "script": "37_open_gap_atr.py",
        "column": "open_gap_atr_scaled_signed_m1_p1"
    },
    {
        "script": "38_dist_prev20_high_atr.py",
        "column": "dist_prev20_high_atr_scaled_signed_m1_p1"
    },
    {
        "script": "39_dist_prev20_low_atr.py",
        "column": "dist_prev20_low_atr_scaled_signed_m1_p1"
    },
    {
        "script": "40_z_close_minus_ema20.py",
        "column": "z_close_minus_ema20_scaled_signed_m1_p1"
    },
    {
        "script": "41_z_typical_price_30.py",
        "column": "z_typical_price_30_scaled_signed_m1_p1"
    },
    {
        "script": "42_cci_scaled_20.py",
        "column": "cci_scaled_20_scaled_signed_m1_p1"
    },
    {
        "script": "43_reversal_return_5.py",
        "column": "reversal_return_5_scaled_signed_m1_p1"
    },
    {
        "script": "44_rsi_reversion_14.py",
        "column": "rsi_reversion_14_semantic_signed_m1_p1"
    },
    {
        "script": "45_vol_z_20.py",
        "column": "vol_z_20_scaled_signed_m1_p1"
    },
    {
        "script": "46_obv_slope_norm_10_20.py",
        "column": "obv_slope_norm_10_20_semantic_signed"
    },
    {
        "script": "47_ret1_x_relvol20.py",
        "column": "ret1_x_relvol20_scaled_signed_m1_p1"
    },
    {
        "script": "48_mfi_centered_14.py",
        "column": "mfi_centered_14_semantic_signed_m1_p1"
    }
]


def run_feature_script(script_name: str, input_file: Path, output_file: Path) -> bool:
    """Run a feature script and return success status."""
    try:
        result = subprocess.run(
            ["python", script_name, "--input", str(input_file), "--output", str(output_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"  [ERROR] Script failed: {result.stderr[:200]}")
            return False
        
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Script timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"  [ERROR] Exception: {str(e)}")
        return False


def test_feature(feature_column: str, test_output: Path, reference_file: Path) -> dict:
    """Test a feature and return results."""
    
    if not test_output.exists():
        return {
            "status": "SKIP",
            "reason": "Test output file not found",
            "match_pct": 0.0
        }
    
    if not reference_file.exists():
        return {
            "status": "SKIP",
            "reason": "Reference file not found",
            "match_pct": 0.0
        }
    
    # Read files
    test_df = pd.read_csv(test_output)
    ref_df = pd.read_csv(reference_file)
    
    # Check feature exists
    if feature_column not in test_df.columns:
        return {
            "status": "SKIP",
            "reason": f"Column not in test output",
            "match_pct": 0.0
        }
    
    if feature_column not in ref_df.columns:
        return {
            "status": "SKIP",
            "reason": f"Column not in reference",
            "match_pct": 0.0
        }
    
    # Extract values
    ref_values = ref_df[feature_column]
    test_values = test_df[feature_column]
    
    total_rows = min(len(ref_values), len(test_values))
    ref_values = ref_values[:total_rows]
    test_values = test_values[:total_rows]
    
    # Compare
    both_nan = ref_values.isna() & test_values.isna()
    both_valid = ref_values.notna() & test_values.notna()
    
    # Changed tolerance to 3 decimal places: 0.001
    tolerance = 1e-3
    if both_valid.sum() > 0:
        matches = np.abs(ref_values[both_valid] - test_values[both_valid]) < tolerance
        matching_values = matches.sum()
    else:
        matching_values = 0
    
    matching_nans = both_nan.sum()
    total_matches = matching_nans + matching_values
    match_pct = (total_matches / total_rows) * 100
    
    # Determine status
    if match_pct >= 99.0:
        status = "PASS"
    elif match_pct >= 95.0:
        status = "WARN"
    else:
        status = "FAIL"
    
    return {
        "status": status,
        "match_pct": match_pct,
        "total_rows": total_rows,
        "matching_nans": matching_nans,
        "matching_values": matching_values,
        "both_valid": both_valid.sum()
    }


def main():
    # Configuration
    INPUT_FILE = Path("../../dataset-v1/ADAUSDT.csv")
    REFERENCE_FILE = Path("../../dataset-v1/ADAUSDT_with_features.csv")
    TEST_OUTPUT = Path("test_output.csv")
    
    print("="*80)
    print("AUTOMATED FEATURE TESTING SUITE")
    print("="*80)
    print(f"Input file:      {INPUT_FILE}")
    print(f"Reference file:  {REFERENCE_FILE}")
    print(f"Test output:     {TEST_OUTPUT}")
    print(f"Total features:  {len(FEATURES)}")
    print(f"Tolerance:       0.001 (3 decimal places)")
    print("="*80)
    print()
    
    # Check input files exist
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return
    
    if not REFERENCE_FILE.exists():
        print(f"[ERROR] Reference file not found: {REFERENCE_FILE}")
        return
    
    # Results tracking
    results = []
    
    # Process each feature
    for idx, feature in enumerate(FEATURES, 1):
        script = feature["script"]
        column = feature["column"]
        
        print(f"[{idx}/{len(FEATURES)}] Processing {script}...")
        print(f"  Column: {column}")
        
        # Run the feature script
        print(f"  Running script...")
        success = run_feature_script(script, INPUT_FILE, TEST_OUTPUT)
        
        if not success:
            results.append({
                "feature": column,
                "script": script,
                "status": "ERROR",
                "match_pct": 0.0
            })
            print(f"  [ERROR] Failed to run script")
            print()
            continue
        
        print(f"  [OK] Script completed")
        
        # Test the feature
        print(f"  Testing feature...")
        test_result = test_feature(column, TEST_OUTPUT, REFERENCE_FILE)
        
        results.append({
            "feature": column,
            "script": script,
            **test_result
        })
        
        # Print result
        status = test_result["status"]
        match_pct = test_result.get("match_pct", 0.0)
        
        if status == "PASS":
            print(f"  [PASS] {match_pct:.2f}% match")
        elif status == "WARN":
            print(f"  [WARN] {match_pct:.2f}% match")
        elif status == "FAIL":
            print(f"  [FAIL] {match_pct:.2f}% match")
        else:
            print(f"  [SKIP] {test_result.get('reason', 'Unknown')}")
        
        print()
        
        # Clean up test output for next iteration
        if TEST_OUTPUT.exists():
            TEST_OUTPUT.unlink()
    
    # Summary report
    print("="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    
    print(f"Total features:  {len(results)}")
    print(f"  PASS:          {passed}")
    print(f"  WARN:          {warned}")
    print(f"  FAIL:          {failed}")
    print(f"  ERROR:         {errors}")
    print(f"  SKIP:          {skipped}")
    print()
    
    # Detailed results table
    print(f"{'Feature':<50} {'Status':<8} {'Match %':<10}")
    print("-"*80)
    for r in results:
        feature_short = r["feature"][:48]
        status = r["status"]
        match_pct = r.get("match_pct", 0.0)
        print(f"{feature_short:<50} {status:<8} {match_pct:>7.2f}%")
    
    print("="*80)
    
    # Final verdict
    if failed > 0 or errors > 0:
        print("[FAILED] Some features did not pass testing")
    elif warned > 0:
        print("[WARNING] All features ran but some have warnings")
    else:
        print("[SUCCESS] All features passed testing!")
    
    print()


if __name__ == "__main__":
    main()
