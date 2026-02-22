# Feature Engineering Implementation Guideline (OHLCV-Only, Leakage-Safe, Fuzzy-Ready)

This document is the single source of truth for building the 60 designed features consistently across all teammates.

## 1) Scope and Objective
1. Build features only from `datetime_utc, symbol, open, high, low, close, volume`.
2. Ensure features are cross-symbol generalizable, scale-invariant, robust to shocks, and fuzzy-ready.
3. Enforce strict no-leakage in both feature computation and normalization.
4. Use feature-specific processing classes (A/B/C), not a one-size-fits-all normalization.
5. Append final feature columns to dataset; do not overwrite core OHLCV columns.

## 2) Non-Negotiable Rules
1. Never use absolute price levels as model features.
2. Never use fixed price-unit thresholds (e.g., “if close moved $100”).
3. Use relative forms: ratios, percentages, ATR-scaled distances, rolling ranks, normalized positions.
4. Never use future candles for any computation.
5. Never fit normalization parameters on full data for offline modeling.
6. Never apply Gaussian central clipping to Class B or Class C features.

## 3) Required Input Contract
1. Required columns: `datetime_utc, symbol, open, high, low, close, volume`.
2. Data must be sorted by `symbol` then `datetime_utc` ascending before feature logic.
3. Duplicate `(symbol, datetime_utc)` rows must be resolved before feature generation.
4. Missing OHLCV rows can exist, but all rolling logic must remain causal and symbol-local.

## 4) Processing Classes (A/B/C)
| Class | Meaning | z/robust z | Gaussian 60% clip | Final scale |
|---|---|---|---|---|
| A | Open-domain, shock-sensitive | Yes (prefer robust z) | Yes (`[-0.8416, +0.8416]`) | Signed `[-1,1]` or Positive `[0,1]` |
| B | Bounded semantic oscillator | No | No | Semantic transform to `[-1,1]` or `[0,1]` |
| C | Binary / categorical / flags | No | No | Keep discrete (`0/1` or `-1/0/1`) |

## 5) Leakage Prevention Standard
1. All rolling stats must be past-only by shifting source series by 1 before rolling.
2. Feature computation must run independently per `symbol`.
3. For offline train/val/test:
1. Fit normalization parameters on train only.
2. Reuse train parameters for val/test.
4. For live/strict temporal mode:
1. Use expanding or rolling past-only normalization.
2. Never include current observation in its own normalization window.
5. Never mix symbols in rolling normalization windows.

## 6) Class A Normalization Pipeline (Mandatory)
For each Class A feature `x_t`:
1. Compute raw relative feature causally.
2. Compute robust z-score (preferred):  
   `rz_t = 0.67448975 * (x_t - median_past) / (MAD_past + eps)`
3. Clip:  
   `rz_clip_t = clip(rz_t, -0.8416, +0.8416)`
4. Scale using fixed bounds only:
1. Signed feature: `scaled_t = rz_clip_t / 0.8416` giving `[-1,1]`
2. Positive feature: `scaled_t = (rz_clip_t + 0.8416) / (2*0.8416)` giving `[0,1]`
5. Do not use dataset min/max scaling.

## 7) Class B Processing Pipeline (Mandatory)
1. No z-score, no robust z-score, no Gaussian clipping.
2. Convert to semantic relative form directly.
3. Examples:
1. RSI: `(RSI - 50)/50` to `[-1,1]`
2. %B: `2*(%B - 0.5)` to `[-1,1]`
3. Stochastic K: `(K - 50)/50` to `[-1,1]`
4. Optional mild clipping (e.g., `[-0.95,0.95]`) only if explicitly justified.

## 8) Class C Processing Pipeline (Mandatory)
1. Keep as discrete values only.
2. No normalization, no clipping.
3. Allowed encodings: `0/1`, or `-1/0/1` for directional flags.

## 9) Handling Warm-Up and Missing Values
1. Expect NaNs at beginning due to EMA/ATR/rolling windows.
2. Keep NaNs during feature generation; do not forward-fill feature values blindly.
3. Downstream model pipeline decides row dropping/imputation policy.
4. Use small epsilon in divisions to avoid instability, not to hide bad data quality.

## 10) Naming Convention (Required)
1. Use lowercase snake_case.
2. Include key parameters in name, e.g. `ema_gap_atr_20`, `atr_ratio_14_63`.
3. Include final semantic suffix in exported column when useful, e.g. `_scaled_signed_m1_p1`, `_scaled_pos_0_1`, `_flag`.
4. Do not create duplicates with different names.

## 11) Implementation Pattern (Pandas/Numpy)
1. Group by `symbol` with `sort=False`, but sort each group by `datetime_utc` inside computation.
2. Build raw feature per group.
3. Apply class-specific transform per group.
4. Concatenate groups and restore original row index order.
5. Append only final output column(s) unless debug mode is explicitly enabled.

## 12) Mandatory Metadata per Feature
For each implemented feature, store metadata in a registry file (`feature_registry.csv` or `.json`):
1. `feature_name`
2. `class` (A/B/C)
3. `raw_formula`
4. `domain_type` (`signed`, `positive`, `bounded`, `flag`)
5. `lookbacks`
6. `normalization_method`
7. `clip_policy`
8. `final_scale`
9. `leakage_notes`
10. `owner` and `version`

## 13) Quality Gates Before Merging
1. Causality test: verify no future reference.
2. Symbol isolation test: verify per-symbol outputs unaffected by other symbols.
3. Domain test:
1. Signed scaled columns stay in `[-1,1]`
2. Positive scaled columns stay in `[0,1]`
3. Flags remain discrete.
4. Shock test: inject extreme candle; confirm Class A clipping limits extremes.
5. Train/val/test test: ensure val/test normalization does not refit.
6. Duplicate logic test: no duplicate feature meaning under new names.

## 14) Common Failure Modes to Avoid
1. Applying z-score to RSI/Stochastic/%B (wrong for Class B).
2. Applying Gaussian clipping to flags (wrong for Class C).
3. Rolling mean/std without `shift(1)` (leakage).
4. Global normalization across all symbols (cross-symbol contamination).
5. Min/max scaling from whole dataset (forbidden).
6. Double-standardizing a feature already defined as z-like.
7. Exporting intermediate debug columns by default.

## 15) Recommended Team Workflow
1. Pick one feature from catalog and confirm its class.
2. Implement raw feature causally per symbol.
3. Apply class-specific processing.
4. Run quality gates.
5. Update feature registry.
6. Submit PR with:
1. code file
2. unit tests
3. one short markdown note containing formula, class, and normalization policy.

## 16) Minimal Unit Test Checklist (Per Feature)
1. Test output column exists with exact expected name.
2. Test row count unchanged.
3. Test index alignment preserved.
4. Test numeric bounds match class policy.
5. Test earliest rows produce expected NaNs from warm-up.
6. Test no dependency on future rows by perturbing future candles and checking current output unchanged.

## 17) Example Policy Mapping for Current Script
1. Feature: `ema_gap_atr_20`
2. Class: A
3. Raw: `(close - EMA20)/ATR14`
4. Normalize: robust z (past-only, per-symbol)
5. Clip: `[-0.8416, +0.8416]`
6. Final scale: signed `[-1,1]`
7. Output style: append final column to original dataset
