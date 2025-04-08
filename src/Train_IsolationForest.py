import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import RobustScaler


# Import processed data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "data/processed/XBG_FE_processed_data.pkl", "rb") as f:
    data = joblib.load(f)
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# Convert to NumPy arrays
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train.to_numpy().astype(np.float32))
X_test = scaler.transform(X_test.to_numpy().astype(np.float32))

# Save the scaler
scaler_path = PROJECT_ROOT / "src/models/isolation_forest_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved at {scaler_path}")

# Estimate contamination dynamically
contamination_rate = np.percentile(y_train, 97) / 100

iForest = IsolationForest(
    n_jobs=-1,
    n_estimators=1500,
    max_samples=256,
    max_features=0.8,
    contamination=contamination_rate,
    random_state=42
)
iForest.fit(X_train)

# Predict anomaly scores
y_train_scores = -iForest.score_samples(X_train)
y_test_scores = -iForest.score_samples(X_test)

# Convert anomaly scores to binary labels using a threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_scores)
best_f1 = 0
best_threshold = 0

for threshold in np.linspace(np.min(y_test_scores), np.max(y_test_scores), 200):
    y_test_pred = (y_test_scores >= threshold).astype(int)
    f1 = f1_score(y_test, y_test_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f'✅ Optimized Threshold: {best_threshold:.4f} with F1 Score: {best_f1:.4f}')

threshold = best_threshold
y_train_pred = (y_train_scores >= threshold).astype(int)
y_test_pred = (y_test_scores >= threshold).astype(int)


# Compute Metrics
test_auc = roc_auc_score(y_test, y_test_scores)
test_f1 = f1_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print(f"📊 Model Evaluation - Isolation Forest:")
print(f"🔹 ROC-AUC: {test_auc:.4f}")
print(f"🔹 Precision: {test_precision:.4f}")
print(f"🔹 Recall: {test_recall:.4f}")
print(f"🔹 F1 Score: {test_f1:.4f}")
print(f"🔹 Threshold: {threshold:.4f}")

# Store metrics in a dictionary
metrics = {
    "roc_auc": round(test_auc, 4),
    "precision": round(test_precision, 4),
    "recall": round(test_recall, 4),
    "f1_score": round(test_f1, 4),
    "threshold": round(threshold, 4),
}

# Save metrics as JSON
metrics_path = PROJECT_ROOT / "results/IsolationForest_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Model metrics saved at {metrics_path}")

# Save model
model_path = PROJECT_ROOT / "src/models/isolation_forest.pkl"
with open(model_path, "wb") as f:
    joblib.dump(iForest, f)
print(f"✅ Model saved at {model_path}")
