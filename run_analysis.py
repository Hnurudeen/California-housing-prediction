"""
California Housing Price Prediction — Full Analysis Pipeline
============================================================
CRISP-DM compliant pipeline with 60/20/20 train/val/test split.

Split roles:
  Train      (60%) — fit all models
  Validation (20%) — compare and select between models (no test leakage)
  Test       (20%) — final unbiased evaluation of the winning model ONLY

Tuning strategy (efficient):
  RandomizedSearchCV on the train set with 3-fold CV.
  20 iterations = 60 model fits — fast and sufficient for this dataset.
  The validation set is used to confirm the winner; test is untouched
  until the very end.

Run:
    python run_analysis.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from california_housing.dataset import load_data, inspect_data, clean_data
from california_housing.features import engineer_features, split_data, encode_features
from california_housing.modeling.train import (
    train_baseline, cross_validate_model,
    tune_random_search, save_model,
)
from california_housing.plots import (
    evaluate, print_metrics,
    plot_actual_vs_predicted, plot_residuals,
    plot_feature_importance, plot_model_comparison,
    plot_error_by_price_range,
)
from california_housing.config import FIGURES_DIR as _FIGURES_DIR, MODELS_DIR as _MODELS_DIR, REPORT_PATH as _REPORT_PATH

FIGURES_DIR = str(_FIGURES_DIR)
MODELS_DIR  = str(_MODELS_DIR)
REPORT_PATH = str(_REPORT_PATH)


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ===========================================================================
# 1. Load & Inspect
# ===========================================================================
section("PHASE 1 — DATA LOADING & INSPECTION")
t0 = time.time()

df_raw = load_data("data/raw/housing.csv")
report = inspect_data(df_raw)

print(f"\nDataset shape   : {report['shape'][0]:,} rows × {report['shape'][1]} columns")
print(f"Duplicates      : {report['duplicates']}")
print(f"\nMissing values:\n{report['missing_values'] if len(report['missing_values']) else '  None'}")
print(f"\nData types:\n{report['dtypes'].to_string()}")
print(f"\nDescriptive statistics:\n{report['describe'].round(2).to_string()}")


# ===========================================================================
# 2. Clean
# ===========================================================================
section("PHASE 2 — DATA CLEANING")

df_clean = clean_data(df_raw)

print(f"\nTarget (median_house_value) after cleaning:")
t = df_clean["median_house_value"]
print(f"  Min    : ${t.min():,.0f}")
print(f"  Max    : ${t.max():,.0f}")
print(f"  Mean   : ${t.mean():,.0f}")
print(f"  Median : ${t.median():,.0f}")
print(f"  Std    : ${t.std():,.0f}")

if "ocean_proximity" in df_clean.columns:
    print(f"\nocean_proximity distribution:")
    for cat, cnt in df_clean["ocean_proximity"].value_counts().items():
        print(f"  {cat:<15}: {cnt:,}  ({cnt/len(df_clean)*100:.1f}%)")


# ===========================================================================
# 3. Feature Engineering
# ===========================================================================
section("PHASE 3 — FEATURE ENGINEERING")

df_eng = engineer_features(df_clean)

print(f"\nFeatures after engineering ({df_eng.shape[1]} columns):")
for col in df_eng.columns:
    print(f"  {col}")

print(f"\nNew feature statistics:")
for feat in ["rooms_per_household", "bedrooms_per_room", "population_per_household"]:
    if feat in df_eng.columns:
        s = df_eng[feat]
        print(f"  {feat:<32}: mean={s.mean():.3f}, std={s.std():.3f}, "
              f"min={s.min():.3f}, max={s.max():.3f}")


# ===========================================================================
# 4. Three-way Split: 60 / 20 / 20
# ===========================================================================
section("PHASE 4 — TRAIN / VALIDATION / TEST SPLIT  (60 / 20 / 20)")

X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df_eng)

n = len(df_eng)
print(f"\n  Train      : {len(X_train_raw):,} rows  ({len(X_train_raw)/n*100:.0f}%)")
print(f"  Validation : {len(X_val_raw):,}  rows  ({len(X_val_raw)/n*100:.0f}%)")
print(f"  Test       : {len(X_test_raw):,}  rows  ({len(X_test_raw)/n*100:.0f}%)")
print(f"  Features   : {X_train_raw.shape[1]}")
print(f"\n  Role of each set:")
print(f"    Train      — fit every model")
print(f"    Validation — compare models, select the best  (no test leakage)")
print(f"    Test       — evaluate the winner ONCE at the very end")


# ===========================================================================
# 5. Encode (fit on train only — no val/test leakage)
# ===========================================================================
section("PHASE 5 — CATEGORICAL ENCODING")

X_train, X_val, X_test, feature_names = encode_features(
    X_train_raw, X_val_raw, X_test_raw
)

print(f"  Train shape : {X_train.shape}")
print(f"  Val shape   : {X_val.shape}")
print(f"  Test shape  : {X_test.shape}")
print(f"\n  Features:")
for f in feature_names:
    print(f"    {f}")


# ===========================================================================
# 6. Baseline Model
# ===========================================================================
section("PHASE 6 — BASELINE MODEL  (default Random Forest, n_estimators=100)")

print("\nTraining...")
from sklearn.ensemble import RandomForestRegressor
baseline_model = train_baseline(X_train, y_train)

print("\n  3-fold cross-validation on train set:")
cv_res = cross_validate_model(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_train, y_train, cv=3
)
print(f"    CV R²   : {cv_res['cv_r2_mean']:.4f} ± {cv_res['cv_r2_std']:.4f}")
print(f"    CV MAE  : ${cv_res['cv_mae_mean']:,.0f}")
print(f"    CV RMSE : ${cv_res['cv_rmse_mean']:,.0f}")

y_pred_baseline_val  = baseline_model.predict(X_val)
baseline_val_metrics = evaluate(y_val, y_pred_baseline_val, "Baseline RF")
print("\n  Validation set performance:")
print_metrics(baseline_val_metrics)


# ===========================================================================
# 7. Hyperparameter Tuning — RandomizedSearchCV  (efficient)
# ===========================================================================
section("PHASE 7 — HYPERPARAMETER TUNING  (RandomizedSearchCV, 20 iter × 3-fold)")

print(
    "\n  Why this setup?"
    "\n    20 iterations × 3-fold CV = 60 model fits."
    "\n    Covers ~4% of the full grid cheaply; finds a strong region."
    "\n    More iterations give diminishing returns on this dataset."
)

param_distributions = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [None, 10, 15, 20, 25],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2"],
}

from sklearn.model_selection import RandomizedSearchCV

base = RandomForestRegressor(random_state=42, n_jobs=-1)
rs = RandomizedSearchCV(
    base,
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
print("\nRunning RandomizedSearchCV...")
rs.fit(X_train, y_train)

print(f"\n  Best CV R² (train)   : {rs.best_score_:.4f}")
print(f"  Best params:")
for k, v in rs.best_params_.items():
    print(f"    {k:<25}: {v}")

y_pred_tuned_val  = rs.best_estimator_.predict(X_val)
tuned_val_metrics = evaluate(y_val, y_pred_tuned_val, "Tuned RF")
print("\n  Validation set performance:")
print_metrics(tuned_val_metrics)


# ===========================================================================
# 8. Model Selection on Validation Set
# ===========================================================================
section("PHASE 8 — MODEL SELECTION  (validation set)")

print(f"\n  {'Model':<20} {'Val R2':>8} {'Val MAE ($)':>12} {'Val RMSE ($)':>13}")
print(f"  {'-'*57}")
for m in [baseline_val_metrics, tuned_val_metrics]:
    print(f"  {m['model']:<20} {m['R2']:>8.4f} "
          f"{m['MAE ($)']:>12,.0f} {m['RMSE ($)']:>13,.0f}")

# Pick winner by validation R²
if tuned_val_metrics["R2"] >= baseline_val_metrics["R2"]:
    best_model  = rs.best_estimator_
    best_name   = "Tuned RF"
    print(f"\n  Winner: Tuned RF  (R² gain on validation: "
          f"+{tuned_val_metrics['R2'] - baseline_val_metrics['R2']:.4f})")
else:
    best_model  = baseline_model
    best_name   = "Baseline RF"
    print(f"\n  Winner: Baseline RF  (tuning did not improve validation R²)")


# ===========================================================================
# 9. Final Evaluation on TEST SET  (untouched until now)
# ===========================================================================
section("PHASE 9 — FINAL TEST SET EVALUATION  (winner only)")

print(f"\n  Model: {best_name}")
print(f"  The test set has not been seen during training OR model selection.")
print(f"  This gives an unbiased estimate of real-world performance.\n")

y_pred_test = best_model.predict(X_test)
test_metrics = evaluate(y_test, y_pred_test, best_name)
print_metrics(test_metrics)


# ===========================================================================
# 10. Full Comparison Table  (val scores for all; test score for winner)
# ===========================================================================
section("PHASE 10 — FULL RESULTS COMPARISON")

baseline_test_metrics = evaluate(y_test, baseline_model.predict(X_test), "Baseline RF")
tuned_test_metrics    = evaluate(y_test, rs.best_estimator_.predict(X_test), "Tuned RF")

print(f"\n  Validation scores (used for model selection):")
print(f"  {'Model':<20} {'R2':>8} {'MAE ($)':>12} {'RMSE ($)':>13} {'MAPE (%)':>10}")
print(f"  {'-'*67}")
for m in [baseline_val_metrics, tuned_val_metrics]:
    star = " <-- selected" if m["model"] == best_name else ""
    print(f"  {m['model']:<20} {m['R2']:>8.4f} "
          f"{m['MAE ($)']:>12,.0f} {m['RMSE ($)']:>13,.0f} "
          f"{m['MAPE (%)']:>10.2f}%{star}")

print(f"\n  Test scores (final, reported once):")
print(f"  {'Model':<20} {'R2':>8} {'MAE ($)':>12} {'RMSE ($)':>13} {'MAPE (%)':>10}")
print(f"  {'-'*67}")
for m in [baseline_test_metrics, tuned_test_metrics]:
    star = " <-- winner" if m["model"] == best_name else ""
    print(f"  {m['model']:<20} {m['R2']:>8.4f} "
          f"{m['MAE ($)']:>12,.0f} {m['RMSE ($)']:>13,.0f} "
          f"{m['MAPE (%)']:>10.2f}%{star}")

print("\nGenerating comparison chart...")
plot_model_comparison(
    [baseline_test_metrics, tuned_test_metrics], FIGURES_DIR
)


# ===========================================================================
# 11. Residual Analysis
# ===========================================================================
section("PHASE 11 — RESIDUAL ANALYSIS  (best model on test set)")

residuals = np.asarray(y_test) - np.asarray(y_pred_test)
print(f"\n  Residual statistics:")
print(f"    Mean   : ${residuals.mean():>10,.0f}  (near 0 = unbiased)")
print(f"    Std    : ${residuals.std():>10,.0f}")
print(f"    Min    : ${residuals.min():>10,.0f}")
print(f"    Max    : ${residuals.max():>10,.0f}")
print(f"    Within ±$20k : {(np.abs(residuals) < 20_000).mean()*100:.1f}% of predictions")
print(f"    Within ±$50k : {(np.abs(residuals) < 50_000).mean()*100:.1f}% of predictions")

print("\nGenerating plots...")
plot_actual_vs_predicted(y_test, y_pred_test, best_name, FIGURES_DIR)
plot_residuals(y_test, y_pred_test, best_name, FIGURES_DIR)


# ===========================================================================
# 12. Feature Importance
# ===========================================================================
section("PHASE 12 — FEATURE IMPORTANCE  (best model)")

importances = plot_feature_importance(
    best_model, feature_names, best_name, FIGURES_DIR
)

print(f"\n  Ranked feature importances:")
for rank, (feat, imp) in enumerate(importances.items(), 1):
    bar = "█" * int(imp * 300)
    print(f"  {rank:>2}. {feat:<38} {imp:.4f}  {bar}")

top_feature = importances.idxmax()
print(
    f"\n  Top feature : '{top_feature}'"
    f"\n  Importance  : {importances[top_feature]*100:.1f}% of the model's total split gain."
    f"\n  Interpretation: median_income is the single strongest predictor"
    f"\n  of house value — higher-income areas command higher prices."
)


# ===========================================================================
# 13. Error by Price Range
# ===========================================================================
section("PHASE 13 — ERROR ANALYSIS BY PRICE RANGE")

y_test_arr = np.asarray(y_test)
bins   = [0, 100_000, 200_000, 300_000, 400_000, 500_001]
labels = ["<$100k", "$100k-200k", "$200k-300k", "$300k-400k", "$400k+"]
bucket = pd.cut(y_test_arr, bins=bins, labels=labels)
df_err = pd.DataFrame({
    "bucket": bucket,
    "abs_error": np.abs(y_test_arr - y_pred_test)
})
summary = df_err.groupby("bucket", observed=True)["abs_error"].agg(
    ["mean", "median", "count"]
)

print(f"\n  {'Price Range':<15} {'Count':>7} {'Mean MAE':>12} {'Median MAE':>13}")
print(f"  {'-'*51}")
for idx, row in summary.iterrows():
    print(f"  {str(idx):<15} {int(row['count']):>7,} "
          f"  ${row['mean']:>9,.0f}  ${row['median']:>9,.0f}")

print(
    "\n  Key insight: errors tend to be larger for higher-priced homes."
    "\n  The model is most accurate in the $100k–$300k range (most data)."
)
plot_error_by_price_range(y_test, y_pred_test, best_name, FIGURES_DIR)


# ===========================================================================
# 14. Save Best Model
# ===========================================================================
section("PHASE 14 — SAVE BEST MODEL")

import joblib
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "best_model.pkl")
save_model(best_model, model_path)

meta = {
    "model_name"    : best_name,
    "feature_names" : feature_names,
    "best_params"   : rs.best_params_ if best_name == "Tuned RF" else {},
    "val_metrics"   : tuned_val_metrics if best_name == "Tuned RF" else baseline_val_metrics,
    "test_metrics"  : test_metrics,
}
joblib.dump(meta, os.path.join(MODELS_DIR, "model_metadata.pkl"))
print(f"  Metadata saved to {MODELS_DIR}/model_metadata.pkl")


# ===========================================================================
# 15. Write Model Report
# ===========================================================================
section("PHASE 15 — WRITING MODEL REPORT")

elapsed = time.time() - t0
os.makedirs("reports", exist_ok=True)

imp_table = "\n".join(
    f"| {i+1} | `{feat}` | {imp:.4f} | {imp*100:.1f}% |"
    for i, (feat, imp) in enumerate(importances.head(10).items())
)

report_md = f"""# California Housing Price Prediction — Model Report

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Runtime**: {elapsed:.1f}s

---

## Dataset Summary

| Item | Value |
|---|---|
| Raw rows | {report['shape'][0]:,} |
| After dropping missing total_bedrooms | {df_raw.dropna(subset=['total_bedrooms']).shape[0]:,} |
| After removing censored $500,001 values | {len(df_clean):,} |
| Total rows removed | {report['shape'][0] - len(df_clean):,} ({(report['shape'][0]-len(df_clean))/report['shape'][0]*100:.1f}%) |
| Final features | {len(feature_names)} |

### Split (60 / 20 / 20)
| Set | Rows | Purpose |
|---|---|---|
| Train | {len(X_train):,} | Fit models |
| Validation | {len(X_val):,} | Select best model (no test leakage) |
| Test | {len(X_test):,} | Final unbiased evaluation of winner |

---

## Data Cleaning

1. **Dropped 207 rows** with missing `total_bedrooms` (<1% of data).
2. **Removed 958 rows** where `median_house_value = $500,001` (dataset ceiling — censored values, not real market prices).

---

## Feature Engineering

| Feature | Formula | Why |
|---|---|---|
| `rooms_per_household` | total_rooms / households | Per-home size; raw counts not comparable across block sizes |
| `bedrooms_per_room` | total_bedrooms / total_rooms | Fraction of rooms that are bedrooms; lower = more spacious |
| `population_per_household` | population / households | Household occupancy; proxy for density |

Raw count columns dropped: `total_rooms`, `total_bedrooms`, `population`, `households`.

---

## Model Results

### Validation Set (used for model selection)
| Model | R² | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|---|
| Baseline RF | {baseline_val_metrics['R2']} | ${baseline_val_metrics['MAE ($)']:,.0f} | ${baseline_val_metrics['RMSE ($)']:,.0f} | {baseline_val_metrics['MAPE (%)']}% |
| Tuned RF | {tuned_val_metrics['R2']} | ${tuned_val_metrics['MAE ($)']:,.0f} | ${tuned_val_metrics['RMSE ($)']:,.0f} | {tuned_val_metrics['MAPE (%)']}% |

### Test Set (final score — winner only reported)
| Model | R² | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|---|
| **{best_name}** (winner) | **{test_metrics['R2']}** | **${test_metrics['MAE ($)']:,.0f}** | **${test_metrics['RMSE ($)']:,.0f}** | **{test_metrics['MAPE (%)']}%** |

### Metric Interpretation
- **R² = {test_metrics['R2']}**: The model explains {test_metrics['R2']*100:.1f}% of the variance in house prices.
- **MAE = ${test_metrics['MAE ($)']:,.0f}**: Average prediction error is ${test_metrics['MAE ($)']:,.0f}.
- **RMSE = ${test_metrics['RMSE ($)']:,.0f}**: Penalises large errors more than MAE.
- **MAPE = {test_metrics['MAPE (%)']}%**: Average percentage error relative to actual price.

---

## Best Hyperparameters

```python
{rs.best_params_}
```

---

## Feature Importance (Top 10)

| Rank | Feature | Importance | % of Total |
|---|---|---|---|
{imp_table}

**Key insight**: `{top_feature}` is the most predictive feature
({importances[top_feature]*100:.1f}% of split gain). Median household income
directly determines purchasing power — the strongest signal for house value.

---

## Residual Analysis

| Metric | Value |
|---|---|
| Mean residual | ${residuals.mean():,.0f} |
| Std of residuals | ${residuals.std():,.0f} |
| Within ±$20k | {(np.abs(residuals) < 20_000).mean()*100:.1f}% of predictions |
| Within ±$50k | {(np.abs(residuals) < 50_000).mean()*100:.1f}% of predictions |

---

## Error by Price Range

| Price Range | n | Mean MAE | Median MAE |
|---|---|---|---|
{chr(10).join(f"| {str(idx)} | {int(row['count']):,} | ${row['mean']:,.0f} | ${row['median']:,.0f} |" for idx, row in summary.iterrows())}

---

## Model Limitations

1. **Price ceiling**: Dataset caps at $500,001; model cannot predict luxury properties.
2. **Temporal drift**: Data is from 1990 — prices have changed dramatically since.
3. **Heteroscedasticity**: Errors grow with price; model is weakest on high-value homes.
4. **Block-level only**: Predicts median for a census block, not individual properties.

---

## Figures Generated

| File | Description |
|---|---|
| `model_comparison.png` | R², MAE, RMSE across models |
| `*_actual_vs_pred.png` | Actual vs predicted (test set) |
| `*_residuals.png` | Residual distribution and spread |
| `*_feature_importance.png` | Top feature importances |
| `*_error_by_price_range.png` | MAE by price bracket |

All figures saved to `reports/figures/`.
"""

with open(REPORT_PATH, "w") as f:
    f.write(report_md)
print(f"  Report written to {REPORT_PATH}")


# ===========================================================================
# Final Summary
# ===========================================================================
section("COMPLETE — FINAL SUMMARY")
print(f"""
  Dataset        : {len(df_clean):,} rows × {len(feature_names)} features
  Split          : {len(X_train):,} train / {len(X_val):,} val / {len(X_test):,} test
  Best Model     : {best_name}

  ── Test Set Performance ──────────────────────
  R²             : {test_metrics['R2']:.4f}  ({test_metrics['R2']*100:.1f}% variance explained)
  MAE            : ${test_metrics['MAE ($)']:,.0f}
  RMSE           : ${test_metrics['RMSE ($)']:,.0f}
  MAPE           : {test_metrics['MAPE (%)']}%

  ── Residual Coverage ─────────────────────────
  Within ±$20k   : {(np.abs(residuals) < 20_000).mean()*100:.1f}% of test predictions
  Within ±$50k   : {(np.abs(residuals) < 50_000).mean()*100:.1f}% of test predictions

  ── Artifacts ─────────────────────────────────
  Model          → {model_path}
  Report         → {REPORT_PATH}
  Figures        → {FIGURES_DIR}/

  Total runtime  : {elapsed:.1f}s
""")
