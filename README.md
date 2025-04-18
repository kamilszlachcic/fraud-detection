# Fraud Detection Project

## Dataset
This project uses a dataset from Kaggle's **IEEE-CIS Fraud Detection** competition.  
Dataset link: [IEEE Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The raw input files `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, and `test_identity.csv` are treated as exports from card transaction monitoring systems. The files are processed, merged, and feature-engineered using modular pipelines.

## Architecture Overview

This project is now structured as a **full MLOps-ready pipeline**, with:

- Reproducible training via `training_pipeline.py`
- Modularized preprocessing & feature engineering
- Feature selection based on XGBoost importance
- Final model calibration using `CalibratedClassifierCV`
- Centralized storage of artifacts (models, features, scalers, metrics)

## Models & Evaluation

The following models were trained and benchmarked:

- ✅ **XGBoost** (Optuna-tuned, early stopping, calibrated)
- ✅ **LightGBM**
- ✅ **Neural Network (PyTorch)**
- ✅ **Autoencoder**
- ✅ **Isolation Forest**

Performance metrics for each model (AUC, F1, Precision, Recall) are saved in `results/*.json`.

## REST API (FastAPI)

A REST API is deployed with two endpoints:

- `POST /predict_raw_files/` — Accepts two uploaded CSVs (`test_transaction.csv` and `test_identity.csv`), processes them end-to-end, and returns fraud predictions.
- `GET /predict_input_processed/` — Returns predictions from a previously saved `.pkl` file with preprocessed input.

This distinction allows easy integration with:
- 🔄 real-time transactional systems (via raw CSV)
- 🧪 saved batch data for evaluation/testing

All predictions are automatically logged to `logs/predictions.log.jsonl` for traceability.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training pipeline:

```bash
python src/pipeline/training_pipeline.py
```

3. Run REST API (requires Docker):

```bash
docker-compose up --build
```

## Results

- All model scores and feature importance are saved in `results/`
- Final production-ready model is saved in `models/XGB_model.pkl`
- Scaler and selected features are saved in `data/processed/`

## References

- **Kaggle Competition**: [IEEE Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
