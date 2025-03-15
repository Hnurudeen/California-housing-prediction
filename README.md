# California Housing Price Prediction with Random Forest 🏠🌲

![Feature Importance Plot]()  


##Data Source
The data is gotten from Kaggle
-[Download Here](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

## 📝 Description  
This project automates real estate valuation by predicting median house prices in California using **Random Forest Regression** and **hyperparameter tuning**. It addresses the limitations of manual appraisals by delivering fast, data-driven predictions with **82.6% accuracy (R² score)**.  

### Key Features:
- **Automated Valuation**: Predicts prices using features like income, location, and housing demographics.
- **Hyperparameter Tuning**: Optimizes model performance with `GridSearchCV` and `RandomizedSearchCV`.
- **Visual Insights**: Generates EDA reports with Sweetviz and feature importance plots.

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
