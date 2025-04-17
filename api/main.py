from fastapi import FastAPI
from typing import List, Dict, Union
import pandas as pd
from pydantic import BaseModel, Field


from api.model_loader import load_model

app = FastAPI(
    title="Fraud Detection API 🚨",
    description="Predicts whether a transaction is fraudulent (0/1) based on tabular features.",
    version="1.0.0",
)

# Load XGBoost model, calibrator, and feature names
model, calibrator, feature_names = load_model()

# ✅ Pydantic schema with dynamic example
class Transaction(BaseModel):
    data: Union[
        Dict[str, float],
        List[Dict[str, float]]
    ] = Field(
        ...,
        examples=[
            {
                feat: 0.0 for feat in feature_names[:10]  # możesz zwiększyć liczbę cech
            }
        ]
    )

@app.post("/predict/")
def predict(transaction: Transaction):
    input_data = transaction.data

    # Support both single and batch prediction
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = pd.DataFrame(input_data)

    # Reorder and fill missing columns
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict calibrated probabilities and binary labels
    probs = calibrator.predict_proba(input_df)[:, 1]
    preds = (probs >= 0.15).astype(int)

    results = [
        {"fraud_prediction": int(p), "probability": round(float(prob), 4)}
        for p, prob in zip(preds, probs)
    ]

    return {"results": results}


@app.get("/features/")
def get_feature_names():
    """Returns the required feature names used during model training."""
    return {"feature_names": feature_names}

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Fraud Detection API 🚨",
        version="1.0.0",
        description="Predicts whether a transaction is fraudulent (0/1) based on tabular features.",
        routes=app.routes,
    )
    openapi_schema["paths"]["/predict/"]["post"]["requestBody"]["content"]["application/json"]["example"] = {
        "data": {
            feat: 0.0 for feat in feature_names[:15]
        }
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
