import json

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Import processed data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "data/processed/XBG_FE_processed_data.pkl", "rb") as f:
    data = joblib.load(f)
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# Convert to NumPy arrays
X_train = X_train.to_numpy().astype(np.float32)
X_test = X_test.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)

# LightGBM dataset format
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)

# Model parameters
params = {
    'objective': 'binary',
    'scale_pos_weight': 15,
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 64,
    'max_depth': 8,
    'n_estimators': 1500,
    'feature_fraction': 0.8,
    'random_state': 42,
    'num_threads': -1,
}

# Train model
model = lgb.train(params, d_train, valid_sets=[d_test], callbacks=[lgb.early_stopping(10)])

# Predict
y_test_pred_proba = model.predict(X_test)
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_pred_proba)
optimal_idx = np.argmax((2 * precisions * recalls) / (precisions + recalls + 1e-8))
best_threshold = thresholds[optimal_idx]
y_test_pred = (y_test_pred_proba > best_threshold).astype(int)

# Compute Metrics
test_auc = roc_auc_score(y_test, y_test_pred_proba)
test_f1 = f1_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print(f"📊 Model Evaluation - LightGBM:")
print(f"🔹 ROC-AUC: {test_auc:.4f}")
print(f"🔹 Precision: {test_precision:.4f}")
print(f"🔹 Recall: {test_recall:.4f}")
print(f"🔹 F1 Score: {test_f1:.4f}")

# Store metrics in a dictionary
metrics = {
    "roc_auc": round(test_auc, 4),
    "precision": round(test_precision, 4),
    "recall": round(test_recall, 4),
    "f1_score": round(test_f1, 4),
}

# Save metrics as JSON
metrics_path = PROJECT_ROOT / "results/LightGBM_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Model metrics saved at {metrics_path}")


# Save model
model_path = PROJECT_ROOT / "src/models/lightgbm_model.txt"
model.save_model(str(model_path))
print(f"✅ Model saved at {model_path}")
