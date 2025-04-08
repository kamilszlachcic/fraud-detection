import json
import pickle
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load best hyperparameters from Optuna tuning
with open(PROJECT_ROOT / "results/optuna_best_params.pkl","rb") as f:
    study_best_params = pickle.load(f)

print("✅ Loaded best hyperparameters from Optuna tuning")
print(f"Best parameters: ", study_best_params)

with open(PROJECT_ROOT / "data/processed/XBG_FE_processed_data.pkl", "rb") as f:
    data = joblib.load(f)
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

xgb_model = xgb.XGBClassifier(**study_best_params)

# ✅ Fit Model with Verbose
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

print("✅ XGBoost training complete!")

# Apply probability calibration
calibrator = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
with tqdm(total=1, desc="Calibrating Model", unit="step") as pbar:
    calibrator.fit(X_train, y_train)  # Train the calibration model
    pbar.update(1)  # Update progress after fit()

print("✅ Calibration complete!")

# Get calibrated probabilities
y_proba_calibrated = calibrator.predict_proba(X_test)[:, 1]

print("✅ Applied Post-Training Calibration!")

# Get predictions using threshold 0.4 (adjust if needed)
optimal_threshold = 0.15
y_pred_adjusted = (y_proba_calibrated > optimal_threshold).astype(int)

# Compute Metrics
roc_auc = roc_auc_score(y_test, y_proba_calibrated)
pr_auc = average_precision_score(y_test, y_proba_calibrated)
f1 = f1_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)

# Print Results
print(f"📊 Model Evaluation:")
print(f"🔹 ROC-AUC: {roc_auc:.4f}")
print(f"🔹 Precision: {precision:.4f}")
print(f"🔹 Recall: {recall:.4f}")
print(f"🔹 F1 Score: {f1:.4f}")
print(f"📌 Precision-Recall AUC: {pr_auc:.4f}")

# Store metrics in a dictionary
metrics = {
    "roc_auc": round(roc_auc, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "pr_auc": round(pr_auc, 4),
    "threshold": optimal_threshold,
}

# Save metrics as JSON
metrics_path = PROJECT_ROOT / "results/XGB_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Model metrics saved at {metrics_path}")

# Save model
save_dict = {
    "model": xgb_model,  # Trained XGBoost model
    "calibrator": calibrator,  # ✅ Save the probability calibrator
    "feature_names": X_train.columns.tolist(),  # Ensures correct input order
}

with open(PROJECT_ROOT / "src/models/XGB_model.pkl", "wb") as f:
    pickle.dump(save_dict, f)
print("✅ Model saved!")
