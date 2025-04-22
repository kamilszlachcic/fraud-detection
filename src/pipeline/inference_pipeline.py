from pathlib import Path
import pandas as pd
import joblib
import pickle
import json

from src.data_processing.utils import drop_v_columns, load_scaler_and_scale
from src.data_processing.feature_engineering import preprocess_and_engineer_features
from src.config import XGB_MODEL_PATH, RESULTS_DIR

def load_model_bundle():
    with open(XGB_MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
    return (
        model_bundle["model"],
        model_bundle["calibrator"],
        model_bundle["feature_names"]
    )

# === Inference function ===
def run_inference(df_input: pd.DataFrame):
    """
    Runs the full inference pipeline on a raw input DataFrame.
    Saves predictions and alerts.
    Returns binary predictions and calibrated probabilities.
    """
    print("\n📥 Starting inference pipeline...")

    # 0. Load model bundle
    model, calibrator, feature_names = load_model_bundle()

    # 1. Drop V-columns
    df = drop_v_columns(df_input)

    # 2. Preprocess (fill missing, encode, engineer features)
    df = preprocess_and_engineer_features(df)

    # 3. Scale features using saved scaler
    df = load_scaler_and_scale(df)

    # 4. Select only model-relevant features
    df = df[feature_names]

    print("🎯 Predicting with df.shape =", df.shape)
    print("🎯 df.head():")
    print(df.head())

    print("🎯 NaNs in df:", df.isnull().sum().sum())
    print("🎯 Unique rows:", df.drop_duplicates().shape[0])

    raw_probs = model.predict_proba(df)[:, 1]
    print("🎯 raw_probs[:10]:", raw_probs[:10])
    calib_probs = calibrator.predict_proba(df)[:, 1]
    print("🎯 calib_probs[:10]:", calib_probs[:10])

    # 5. Predict
    probs = calibrator.predict_proba(df)[:, 1]
    preds = (probs >= 0.15).astype(int)

    # 6. Prepare output DataFrame
    output = df_input.copy()
    output["fraud_probability"] = probs
    output["isFraud_pred"] = preds

    # 7. Save full prediction results
    output.to_parquet(RESULTS_DIR / "predictions.parquet", index=False)
    print("📄 Saved full predictions to predictions.parquet")

    # 8. Save alerts (only high-risk predictions)
    alerts = output[output["isFraud_pred"] == 1].copy()
    alerts_out = alerts[["TransactionID", "fraud_probability"]].to_dict(orient="records")
    with open(RESULTS_DIR / "alerts.json", "w") as f:
        json.dump(alerts_out, f, indent=4)
    print(f"🚨 Saved {len(alerts)} alerts to alerts.json")

    print("✅ Inference complete! Returning predictions.")
    return preds, probs
