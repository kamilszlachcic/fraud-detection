from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import io
import json
from datetime import datetime

from src.pipeline.inference_pipeline import run_inference
from src.config import PREDICTION_LOG_PATH, RESULTS_DIR

app = FastAPI(
    title="Fraud Detection API 🚨",
    description="Upload credit card data (CSV) to get fraud predictions",
    version="1.0"
)

# === Helper: log predictions ===
def log_predictions(transaction_ids, preds, probs, source):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PREDICTION_LOG_PATH, "a") as log_file:
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
            {"prediction": int(p),
             "probability": float("{:.6f}".format(prob))}
            for p, prob in zip(preds, probs)
        ]
        return {"results": results[:50]}  # return preview

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get_alerts/")
def get_alerts():
    alerts_path = RESULTS_DIR / "alerts.json"
    if alerts_path.exists():
        with open(alerts_path, "r", encoding="utf-8") as f:
            alerts = json.load(f)
        return {"n_alerts": len(alerts), "alerts": alerts}
    else:
        return JSONResponse(status_code=404, content={"error": "alerts.json not found"})

@app.get("/download/alerts/")
def download_alerts():
    alerts_path = RESULTS_DIR / "alerts.json"
    if alerts_path.exists():
        return FileResponse(alerts_path, media_type='application/json', filename="alerts.json")
    else:
        return JSONResponse(status_code=404, content={"error": "alerts.json not found"})

from src.config import PREDICTION_LOG_PATH

@app.get("/get_predictions/")
def get_predictions():
    try:
        with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f.readlines()[:50]]  # lub f.readlines()[-50:]
        return {
            "n_predictions": len(records),
            "predictions": records
        }
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "predictions.log.jsonl not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
