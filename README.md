
# 🕵️‍♂️ Fraud Detection Project

A full-featured MLOps-ready pipeline for real-world fraud detection on transactional data.  
Designed to be modular, reproducible, interpretable, and production-ready.

---

## 📊 Dataset
This project uses the dataset from Kaggle's **IEEE-CIS Fraud Detection** competition.  
Dataset link: [IEEE Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

---

## 🏗️ Architecture Overview

This repository features a robust pipeline that includes:

- ✅ Reproducible training (`training_pipeline.py`)
- ✅ Modular preprocessing and feature engineering (`src/data_processing/`)
- ✅ Feature selection based on XGBoost importance
- ✅ Hyperparameter tuning with **Optuna**
- ✅ Model calibration using `CalibratedClassifierCV`
- ✅ Auto-saving of metrics, models, scalers, features, and charts
- ✅ MLflow tracking for all experiments and artifacts
- ✅ Containerized REST API (Docker + FastAPI)
- ✅ Versioned code and dependency management with **Poetry**

---

## 🧠 Models & Benchmarks

| Model               | ROC AUC | Precision | Recall | F1 Score | PR AUC |
|--------------------|---------|-----------|--------|----------|--------|
| 🥇 **XGBoost**       | 0.9769  | 0.9599    | 0.707  | 0.8143   | 0.8805 |
| 🥈 **LightGBM**      | 0.9697  | 0.8184    | 0.7218 | 0.7670   |   —    |
| 🧪 **IsolationForest** | 0.7552  | 0.1812    | 0.3429 | 0.2371   |   —    |
| 🔬 **Autoencoder**   | In progress – advanced architecture coming soon |

All metrics are logged per-run to `results/` and via MLflow.

---

## 🔬 MLflow Experiment Tracking

All training runs are tracked using [MLflow](https://mlflow.org/) for complete transparency and reproducibility.

```bash
mlflow ui --port 5000
# Visit: http://127.0.0.1:5000
```

Each run includes:

- 🎯 Hyperparameters (e.g. max_depth=12, learning_rate=0.0173)
- 📊 Metrics (ROC AUC, F1, Precision, Recall, PR AUC)
- 💾 Artifacts: models, feature selectors, visualizations, configs

Example run summary (XGBoost final model):

```
ROC AUC: 0.9769
F1 Score: 0.8143
Precision: 0.9599
Recall: 0.707
PR AUC: 0.8805
Threshold: 0.15
Best iteration: 2499
```

Artifacts:
- `XGB_model.pkl` (calibrated model)
- `feature_importance.pkl`, `top50_feature_importance.png`
- `selected_features.pkl`, `XGB_metrics.json`

---

## 🧪 REST API (FastAPI + Docker)

Containerized REST API for local or cloud deployment.

### Endpoints

- `POST /predict_raw_files/` — accepts `test_transaction.csv` and `test_identity.csv`, returns predictions
- `GET /predict_input_processed/` — returns predictions from preprocessed `.pkl` input

### Usage

```bash
docker-compose up --build
```

API logs all predictions to `logs/predictions.log.jsonl`.

---

## ⚙️ Environment Setup

Install dependencies using Poetry:

```bash
poetry install
```

Train the model:

```bash
python src/pipeline/training_pipeline.py
```

---

## 🔮 What's Next?

This project is actively evolving. Upcoming features:

- 🤖 Deep Autoencoder model for anomaly-based fraud detection
- 📦 Model Registry via MLflow
- 📈 Threshold optimization dashboards
- ☁️ Deployment to Azure ML or AWS Sagemaker
- 🔍 SHAP-based explainability & feature visualization
