# Problem Statement

## Business Context

Residential property valuation in California is an expensive, slow, and
inconsistent process when performed manually by human appraisers. Automated
Valuation Models (AVMs) can provide fast, repeatable estimates from publicly
available census and geographic data.

## Objective

Build a regression model that predicts the **median house value** of a
California census block group using demographic and geographic features
from the 1990 US Census.

## Success Criteria

| Criterion | Target |
|---|---|
| R² on held-out test set | ≥ 0.80 |
| Mean Absolute Error | < $35,000 |
| MAPE | < 20% |

## Constraints

- **No leakage**: Test set must not be used to fit any transformation.
- **Reproducibility**: Fixed `random_state=42` throughout; data pipeline is deterministic.
- **Interpretability**: Feature importances must be reported.

## Dataset

- **Source**: 1990 California census, compiled by Pace & Barry (1997).
  Available on [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
- **Unit of observation**: One census block group (~1,000–3,000 people).
- **Target variable**: `median_house_value` — median value of owner-occupied
  homes in the block (USD, 1990 dollars). Capped at $500,001 in the dataset.

## Known Limitations

1. Data is from 1990; predictions will not reflect modern California prices.
2. The $500,001 ceiling means the model cannot predict luxury home prices.
3. The model predicts median block-group value, not individual property value.