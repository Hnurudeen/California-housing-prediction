"""
Feature engineering, train/val/test splitting, and categorical encoding.

Design decisions:
- Split BEFORE encoding/scaling to prevent data leakage
- Fit encoders on training set only, align val and test to same columns
- Engineer ratio features (more meaningful than raw block-level counts)
- Drop original count columns after engineering ratios
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from california_housing.config import RANDOM_STATE, TEST_SIZE, VAL_SIZE, TARGET_COL


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features from block-level counts.

    Raw counts (total_rooms, total_bedrooms, population, households) are
    not comparable across blocks of different sizes. Normalising by
    households produces per-household ratios that are more informative
    and interpretable.

    New features:
    - rooms_per_household      : average number of rooms per household
    - bedrooms_per_room        : fraction of rooms that are bedrooms
                                  (lower = more living space)
    - population_per_household : average household occupancy
    """
    df = df.copy()

    df["rooms_per_household"]      = df["total_rooms"]    / df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"]     / df["households"]

    # Drop raw count columns — replaced by the ratios above
    df = df.drop(columns=["total_rooms", "total_bedrooms", "population", "households"])

    print(
        "\nFeature Engineering:"
        "\n  Added  : rooms_per_household, bedrooms_per_room, population_per_household"
        "\n  Removed: total_rooms, total_bedrooms, population, households"
    )
    return df


# ---------------------------------------------------------------------------
# Split (BEFORE encoding — prevents data leakage)
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Three-way split: 60% train / 20% validation / 20% test.

    Why three sets?
    - Train (60%)      : used to fit all models.
    - Validation (20%) : used to compare and select between models
                         without touching the test set.
    - Test (20%)       : held out entirely; evaluated ONCE on the
                         winning model. Gives an unbiased final estimate.

    Encoding is applied AFTER this split so no test/val statistics
    leak into training transforms.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off the test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # From the remaining 80%, split off validation (20% of total = 25% of 80%)
    val_fraction_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Encode (fit on train only, transform all three splits)
# ---------------------------------------------------------------------------

def encode_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    One-hot encode categorical features across all three splits.

    Fit on X_train only; align X_val and X_test to the same columns.
    This ensures neither the validation nor test set influences the encoding.

    Random Forest does not require feature scaling, so we skip StandardScaler.
    """
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
        X_val   = pd.get_dummies(X_val,   columns=cat_cols, drop_first=False)
        X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=False)
        # Align val/test to the columns seen in train
        X_val   = X_val.reindex(columns=X_train.columns,  fill_value=0)
        X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
        print(f"\nEncoded categorical columns: {cat_cols}")
    else:
        print("\nNo categorical columns to encode.")

    print(f"  Final feature count: {X_train.shape[1]}")
    feature_names = X_train.columns.tolist()
    return X_train, X_val, X_test, feature_names
