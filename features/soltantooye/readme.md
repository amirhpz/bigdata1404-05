## ✅ Test Report (Feature Validation)

**Command used:**
```bash
python run_all_tests.py
```

### Summary

| Metric | Count |
|---:|---:|
| Total features | 12 |
| PASS | 7 |
| WARN | 0 |
| FAIL | 5 |
| ERROR | 0 |
| SKIP | 0 |

---

### Results (per feature)

| Feature | Status | Match % |
|---|:---:|---:|
| `open_gap_atr_scaled_signed_m1_p1` | PASS | 100.00% |
| `dist_prev20_high_atr_scaled_signed_m1_p1` | PASS | 100.00% |
| `dist_prev20_low_atr_scaled_signed_m1_p1` | PASS | 100.00% |
| `z_close_minus_ema20_scaled_signed_m1_p1` | FAIL | 30.96% |
| `z_typical_price_30_scaled_signed_m1_p1` | FAIL | 37.76% |
| `cci_scaled_20_scaled_signed_m1_p1` | PASS | 100.00% |
| `reversal_return_5_scaled_signed_m1_p1` | PASS | 100.00% |
| `rsi_reversion_14_semantic_signed_m1_p1` | PASS | 100.00% |
| `vol_z_20_scaled_signed_m1_p1` | FAIL | 39.59% |
| `obv_slope_norm_10_20_semantic_signed` | FAIL | 29.05% |
| `ret1_x_relvol20_scaled_signed_m1_p1` | FAIL | 49.00% |
| `mfi_centered_14_semantic_signed_m1_p1` | PASS | 100.00% |

---


- **FAIL** indicates the produced feature values did not match the reference implementation closely enough (see *Match %*).
- No **WARN**, **ERROR**, or **SKIP** cases were reported in this run.

