from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
import joblib
import pickle
import json
from datetime import datetime
from pathlib import Path

from src.pipeline.inference_pipeline import run_inference

app = FastAPI(
    title="Fraud Detection API 🚨",
    description="Upload credit card data (CSV) to get fraud predictions",
    version="1.0"
)

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/XGB_model.pkl"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data/processed/inference_input_processed.pkl"
LOG_PATH = PROJECT_ROOT / "logs"
LOG_PATH.mkdir(exist_ok=True)
PREDICTION_LOG = LOG_PATH / "predictions.log.jsonl"

# === Load model ===
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)


# === Helper: log predictions ===
def log_predictions(transaction_ids, preds, probs, source):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PREDICTION_LOG, "a") as log_file:
        for tid, pred, prob in zip(transaction_ids, preds, probs):
            record = {
                "TransactionID": int(tid) if pd.notna(tid) else None,
                "prediction": int(pred),
                "probability": round(float(prob), 4),
                "source": source,
                "timestamp": now
            }
            log_file.write(json.dumps(record) + "\n")


# === Endpoint: predict from 2 raw files ===
@app.post("/predict_raw_files/")
async def predict_raw_files(
        file_transaction: UploadFile = File(...),
        file_identity: UploadFile = File(...)
):
    try:
        contents_tx = await file_transaction.read()
        contents_id = await file_identity.read()

        df1 = pd.read_csv(io.BytesIO(contents_tx))
        df2 = pd.read_csv(io.BytesIO(contents_id))

        df = df1.merge(df2, how="left", on="TransactionID")
        df.columns = df.columns.str.replace("-", "_")

        preds, probs = run_inference(df)

        # Log predictions
        log_predictions(df["TransactionID"], preds, probs, source="raw_files")

        results = [
            {"prediction": int(p), "probability": round(float(prob), 4)}
            for p, prob in zip(preds, probs)
        ]
        return {"results": results[:50]}  # return preview

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# === Endpoint: predict from inference_input_processed .pkl ===
@app.get("/predict_input_processed/")
def predict_from_input_processed():
    try:
        sample = joblib.load(SAMPLE_DATA_PATH)
        df = sample["X_input"]

        preds, probs = run_inference(df)

        # Log predictions
        log_predictions(df.index if "TransactionID" not in df.columns else df["TransactionID"], preds, probs,
                        source="processed")

        results = [
            {"prediction": int(p), "probability": round(float(prob), 4)}
            for p, prob in zip(preds, probs)
        ]
        return {"results": results[:10]}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
