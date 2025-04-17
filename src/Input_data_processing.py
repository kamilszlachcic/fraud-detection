import pandas as pd
from src.Train_data_processing import apply_magic_features, preprocess_and_engineer_features


def prepare_data_for_prediction(df_raw: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Full preprocessing pipeline for inference time.

    - Applies magic features
    - Handles missing values, encoding, and feature engineering
    - Scales numeric values
    - Reindexes columns to match training feature set
    - Optimizes dtypes
    """
    df_magic = apply_magic_features(df_raw)

    # Use test_size=0.0 to get full transformed dataset
    X_pred, _, _, _ = preprocess_and_engineer_features(df_magic, test_size=0.0)

    # Align with training features
    X_pred = X_pred.reindex(columns=feature_names, fill_value=0)

    # Optimize data types
    X_pred = X_pred.astype({
        col: "int32" for col in X_pred.select_dtypes(include=["int64"]).columns
    })
    X_pred = X_pred.astype({
        col: "float32" for col in X_pred.select_dtypes(include=["float64"]).columns
    })

    return X_pred
