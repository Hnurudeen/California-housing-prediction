"""
Run inference with a saved model.

Usage:
    from california_housing.modeling.predict import load_model, load_metadata, predict_from_raw

    model = load_model()
    meta  = load_metadata()
    predictions = predict_from_raw(df_new, model, meta)
"""

import joblib
import numpy as np
import pandas as pd

from california_housing.config import MODEL_PATH, METADATA_PATH
from california_housing.features import engineer_features


def load_model(model_path=None):
    """Load a saved model from disk. Defaults to models/best_model.pkl."""
    path = model_path or MODEL_PATH
    return joblib.load(path)


def load_metadata(meta_path=None):
    """Load saved model metadata (feature names, metrics, best params)."""
    path = meta_path or METADATA_PATH
    return joblib.load(path)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Run predictions on an already-encoded feature DataFrame."""
    return model.predict(X)


def predict_from_raw(df_raw: pd.DataFrame, model=None, meta=None) -> np.ndarray:
    """
    Full prediction pipeline from raw (pre-cleaned) input features.

    Applies feature engineering and one-hot encoding, aligning columns
    to those seen during training (stored in metadata).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Input rows with the same columns as the original dataset
        (excluding median_house_value).
    model : trained sklearn estimator, optional
        Loaded from models/best_model.pkl if not provided.
    meta : dict, optional
        Loaded from models/model_metadata.pkl if not provided.

    Returns
    -------
    np.ndarray of predicted house values in USD.
    """
    if model is None:
        model = load_model()
    if meta is None:
        meta = load_metadata()

    feature_names = meta["feature_names"]

    df_eng = engineer_features(df_raw)
    df_enc = pd.get_dummies(df_eng, drop_first=False)
    df_enc = df_enc.reindex(columns=feature_names, fill_value=0)

    return model.predict(df_enc)
