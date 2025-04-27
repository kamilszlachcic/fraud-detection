import pandas as pd
import xgboost as xgb
import joblib
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime

import mlflow
import mlflow.xgboost

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

from src.data_processing.utils import (
    load_and_merge,
    drop_v_columns,
    detect_columns_to_scale,
    fit_scaler_and_scale,
    split_train_test
)
from src.data_processing.feature_engineering import preprocess_and_engineer_features
from src.config import MODELS_DIR, DATA_PROCESSING_DIR, DATA_RAW_DIR, RESULTS_DIR


# === Load and preprocess ===
print("\n🚀 [1] Loading and preprocessing raw data...")
df = load_and_merge(
    transaction_file=DATA_RAW_DIR / "train_transaction.csv",
    identity_file=DATA_RAW_DIR / "train_identity.csv"
)
df = drop_v_columns(df)
df = preprocess_and_engineer_features(df)

# === Train/test split ===
print("\n📊 [2] Splitting train/test...")
X_train, X_test, y_train, y_test = split_train_test(df)

# === Feature scaling ===
print("\n⚖️ [3] Scaling features...")
columns_to_scale = detect_columns_to_scale(X_train)
X_train, scaler = fit_scaler_and_scale(X_train, columns_to_scale, save=True)
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

# === Step 1: Feature Importance model ===
print("\n📈 [4] Calculating feature importance using base XGBoost model...")
xgb_default = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=500,
    max_depth=4,
    learning_rate=0.1,
    subsample=1,
    colsample_bytree=1,
    scale_pos_weight=1,
    nthread=-1,
    random_state=42,
    )

xgb_default.fit(X_train, y_train, verbose=True)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_default.feature_importances_
}).sort_values(by="Importance", ascending=False)

# === Apply selection logic
feature_importance["Cumulative_Importance"] = feature_importance["Importance"].cumsum()
cumulative_importance_threshold = 1
if cumulative_importance_threshold >= 1:
    selected_features = feature_importance["Feature"].tolist()
else:
    selected_features = feature_importance[
        feature_importance["Cumulative_Importance"] <= cumulative_importance_threshold
    ]["Feature"].tolist()

print(
    f"✅ Selected {len(selected_features)} features "
    f"(Explaining {cumulative_importance_threshold * 100}% of total importance)"
)

# Save results
joblib.dump(feature_importance, DATA_PROCESSING_DIR / "feature_importance.pkl")
joblib.dump(selected_features, DATA_PROCESSING_DIR / "selected_features.pkl")


plt.figure(figsize=(20, 8))
plt.barh(feature_importance["Feature"].head(50), feature_importance["Importance"].head(50))
plt.gca().invert_yaxis()
plt.tight_layout()
plt.title("Top 50 Feature Importance")
plt.savefig(RESULTS_DIR / "top50_feature_importance.png")
plt.close()

# === Filter features ===
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# === MLflow] Initializing experiment tracking ===
print("\n📘 [MLflow] Initializing experiment tracking...")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_detection_experiment")

mlflow_run_name = f"xgb_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.start_run(run_name=mlflow_run_name)

# === Step 2: Train with Optuna parameters ===
print("\n🏆 [5] Training final model with best Optuna parameters...")

# Load best parameters
with open(RESULTS_DIR / "optuna_best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

best_params.update({
    "nthread": -1,
    "random_state": 42,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50
})

print("✅ Loaded best hyperparameters from Optuna tuning")
print(f"Best parameters: ", best_params)

mlflow.log_params(best_params)

# Step 2.1: Train with early stopping to find best_iteration
temp_model = xgb.XGBClassifier(**best_params)

temp_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)


best_iter = temp_model.best_iteration
print(f"✅ Best iteration found: {best_iter}")

# Step 2.2: Re-train final model with best number of trees

best_params.update({
    "nthread": -1,
    "random_state": 42,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 0,
    "n_estimators": best_iter
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# === Calibration ===
print("\n🎯 [6] Calibrating predictions...")
calibrator = CalibratedClassifierCV(final_model, method="sigmoid", cv="prefit")
calibrator.fit(X_train, y_train)

# === Evaluation ===
print("\n📊 [7] Evaluating model...")
probs = calibrator.predict_proba(X_test)[:, 1]
preds = (probs >= 0.15).astype(int)

metrics = {
    "roc_auc": round(roc_auc_score(y_test, probs), 4),
    "precision": round(precision_score(y_test, preds), 4),
    "recall": round(recall_score(y_test, preds), 4),
    "f1_score": round(f1_score(y_test, preds), 4),
    "pr_auc": round(average_precision_score(y_test, probs), 4),
    "threshold": 0.15,
    "best_iteration": best_iter,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "params": best_params
}

mlflow.log_metrics({
    "roc_auc": metrics["roc_auc"],
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "f1_score": metrics["f1_score"],
    "pr_auc": metrics["pr_auc"]
})


# Save metrics
metrics_path = RESULTS_DIR / "XGB_metrics.json"
if metrics_path.exists():
    try:
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
            if not isinstance(all_metrics, list):
                all_metrics = [all_metrics]
    except json.JSONDecodeError:
        all_metrics = []
else:
    all_metrics = []

all_metrics.append(metrics)

with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=4)

print("✅ Metrics saved.")

# Save model
print("\n💾 [8] Saving model...")
model_artifacts = {
    "model": final_model,
    "calibrator": calibrator,
    "feature_names": X_train.columns.tolist()
}

with open(MODELS_DIR / "XGB_model.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)

mlflow.log_artifact(str(MODELS_DIR / "XGB_model.pkl"))
mlflow.log_artifact(str(DATA_PROCESSING_DIR / "feature_importance.pkl"))
mlflow.log_artifact(str(DATA_PROCESSING_DIR / "selected_features.pkl"))
mlflow.log_artifact(str(RESULTS_DIR / "top50_feature_importance.png"))
mlflow.log_artifact(str(metrics_path))

print("\n🎉 Training pipeline complete!")
mlflow.end_run()