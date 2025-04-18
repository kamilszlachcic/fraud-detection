from pathlib import Path
import pandas as pd
import joblib
import pickle
import json

from src.data_preprocessing.utils import drop_v_columns, load_scaler_and_scale
from src.data_preprocessing.feature_engineering import preprocess_and_engineer_features

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === Load model and metadata ===
with open(MODELS_DIR / "XGB_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
calibrator = model_bundle["calibrator"]
feature_names = model_bundle["feature_names"]

# === Inference function ===
def run_inference(df_input: pd.DataFrame):
    """
    Runs the full inference pipeline on a raw input DataFrame.
    Saves predictions and alerts.
    Returns binary predictions and calibrated probabilities.
    """
    print("\n📥 Starting inference pipeline...")

    # 1. Drop V-columns
    df = drop_v_columns(df_input)

    # 2. Preprocess (fill missing, encode, engineer features)
    df = preprocess_and_engineer_features(df)

    # 3. Scale features using saved scaler
    df = load_scaler_and_scale(df)

    # 4. Select only model-relevant features
    df = df[feature_names]

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
