# California Housing Price Prediction with Random Forest 🏠🌲

##Data Source
The data is gotten from Kaggle
-[Download Here](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

## 📝 Description  
This project automates real estate valuation by predicting median house prices in California using **Random Forest Regression** and **hyperparameter tuning**. It addresses the limitations of manual appraisals by delivering fast, data-driven predictions with **82.6% accuracy (R² score)**.  

### Key Features:
- **Automated Valuation**: The algorithm predicts prices using features like income, location, and housing demographics etc.
- **Hyperparameter Tuning**: Optimizes model performance with `GridSearchCV` and `RandomizedSearchCV`.
- **Visual Insights**: Generated EDA reports with Sweetviz and feature importance plots.

Feature Importance

This shows the most inportant feature in the dataset, from the image it shows that ...
(![Feature Importance](https://github.com/user-attachments/assets/775115f8-4f91-4e99-ad7b-f73744fedffc)


## 📊 Results  
| Metric               | Value                     |
|----------------------|---------------------------|
| **R² Score**         | 0.826 (82.6% variance explained) |
| **Mean Absolute Error (MAE)** | $31,720          |
| **Mean Squared Error (MSE)**  | 2,772,601,365    |

**Best Model**:  
```python
Best hyperparameters: {
  'max_depth': 10,
  'min_samples_leaf': 4,
  'min_samples_split': 10,
  'n_estimators': 800
}
