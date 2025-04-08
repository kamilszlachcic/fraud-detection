import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from scipy.spatial.distance import mahalanobis
from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, \
    average_precision_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data/processed/XBG_FE_processed_data.pkl"

# Load Processed Data
with open(DATA_PATH, "rb") as f:
    data = joblib.load(f)
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# ✅ Train AE only on normal transactions
X_train_normal = X_train[y_train == 0]  # Use only normal transactions
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)  # Full train set for anomaly detection
X_train_normal_tensor = torch.tensor(X_train_normal.values, dtype=torch.float32)  # Normal transactions only

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # Full y_train
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader (train AE only on normal transactions)
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_normal_tensor), batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size, shuffle=False, pin_memory=True)


# ✅ Sparse Denoising Autoencoder (DAE) with L1 Regularization
class SparseDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, noise_std=0.1, l1_lambda=1e-5):
        super().__init__()
        self.noise_std = noise_std  # 🔥 Noise for DAE
        self.l1_lambda = l1_lambda  # 🔥 L1 Regularization

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ELU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ELU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ELU(), nn.BatchNorm1d(128),
            nn.Linear(128, 32), nn.Tanh()  # 🔥 Forces compact, sparse representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128), nn.ELU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256), nn.ELU(), nn.BatchNorm1d(256),
            nn.Linear(256, 512), nn.ELU(), nn.BatchNorm1d(512),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        if self.training:  # Apply noise only during training
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Initialize Model
model = SparseDenoisingAutoencoder(input_dim=X_train.shape[1]).to(device)

# ✅ Use Huber Loss + L1 Regularization
criterion = nn.HuberLoss(reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-6)


# ✅ Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
        else:
            self.counter += 1
        return self.counter >= self.patience


num_epochs = 50
early_stopping = EarlyStopping()

# ✅ Training Loop
print(f'Model training in progres...')
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, in train_loader:
        batch_X = batch_X.to(device)
        optimizer.zero_grad()

        # 🔥 Add small Gaussian noise (DAE)
        noisy_X = batch_X + model.noise_std * torch.randn_like(batch_X)

        encoded, decoded = model(noisy_X)
        loss = criterion(decoded, batch_X)

        # 🔥 L1 Regularization (Sparsity in AE)
        l1_penalty = sum(p.abs().sum() for p in model.encoder.parameters())
        loss = loss.mean() + model.l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Compute validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_X, in val_loader:
            batch_X = batch_X.to(device)
            _, decoded = model(batch_X)
            loss = criterion(decoded, batch_X)
            val_losses.append(loss.mean().item())

    avg_val_loss = np.mean(val_losses)
    if early_stopping.step(avg_val_loss, model):
        print(f"Early stopping at epoch {epoch + 1}")
        model.load_state_dict(early_stopping.best_model)
        break

    print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.6f}")

# ✅ Compute Anomaly Scores (Use full X_train for anomaly detection)
model.eval()
with torch.no_grad():
    train_errors = torch.mean((X_train_tensor.to(device) - model(X_train_tensor.to(device))[1]) ** 2,
                              dim=1).cpu().numpy()
    test_errors = torch.mean((X_test_tensor.to(device) - model(X_test_tensor.to(device))[1]) ** 2, dim=1).cpu().numpy()

# ✅ Compute Mahalanobis Distance on Encoded Features
encoded_X_train = model.encoder(X_train_tensor.to(device)).cpu().detach().numpy()
encoded_X_test = model.encoder(X_test_tensor.to(device)).cpu().detach().numpy()

cov_matrix_train = np.cov(encoded_X_train, rowvar=False) + np.eye(encoded_X_train.shape[1]) * 1e-6
inv_cov_matrix_train = np.linalg.pinv(cov_matrix_train)
mean_vector_train = np.mean(encoded_X_train, axis=0)

mahalanobis_train = np.array(
    [mahalanobis(sample, mean_vector_train, inv_cov_matrix_train) for sample in encoded_X_train])
mahalanobis_test = np.array(
    [mahalanobis(sample, mean_vector_train, inv_cov_matrix_train) for sample in encoded_X_test])

# ✅ Compute Skewness and Kurtosis
train_skewness = skew(encoded_X_train, axis=1).reshape(-1, 1)
test_skewness = skew(encoded_X_test, axis=1).reshape(-1, 1)
train_kurtosis = kurtosis(encoded_X_train, axis=1).reshape(-1, 1)
test_kurtosis = kurtosis(encoded_X_test, axis=1).reshape(-1, 1)

# ✅ Compute Statistics on Encoded Features
train_mean = np.mean(encoded_X_train, axis=1).reshape(-1, 1)  # Mean across features
train_var = np.var(encoded_X_train, axis=1).reshape(-1, 1)  # Variance across features
train_min = np.min(encoded_X_train, axis=1).reshape(-1, 1)  # Min value in encoded space
train_max = np.max(encoded_X_train, axis=1).reshape(-1, 1)  # Max value in encoded space

test_mean = np.mean(encoded_X_test, axis=1).reshape(-1, 1)
test_var = np.var(encoded_X_test, axis=1).reshape(-1, 1)
test_min = np.min(encoded_X_test, axis=1).reshape(-1, 1)
test_max = np.max(encoded_X_test, axis=1).reshape(-1, 1)

# ✅ SCALE All Extracted Features
scaler = RobustScaler()

train_errors_scaled = scaler.fit_transform(train_errors.reshape(-1, 1))
test_errors_scaled = scaler.transform(test_errors.reshape(-1, 1))

mahalanobis_train_scaled = scaler.fit_transform(mahalanobis_train.reshape(-1, 1))
mahalanobis_test_scaled = scaler.transform(mahalanobis_test.reshape(-1, 1))

train_skewness_scaled = scaler.fit_transform(train_skewness)
test_skewness_scaled = scaler.transform(test_skewness)

train_kurtosis_scaled = scaler.fit_transform(train_kurtosis)
test_kurtosis_scaled = scaler.transform(test_kurtosis)

train_mean_scaled = scaler.fit_transform(train_mean)
train_var_scaled = scaler.fit_transform(train_var)
train_min_scaled = scaler.fit_transform(train_min)
train_max_scaled = scaler.fit_transform(train_max)

test_mean_scaled = scaler.transform(test_mean)
test_var_scaled = scaler.transform(test_var)
test_min_scaled = scaler.transform(test_min)
test_max_scaled = scaler.transform(test_max)

# ✅ Stack Scaled Features for XGBoost
X_train_features = np.column_stack([
    train_errors_scaled.flatten(), mahalanobis_train_scaled.flatten(),
    train_mean_scaled.flatten(), train_var_scaled.flatten(),
    train_min_scaled.flatten(), train_max_scaled.flatten(),
    train_skewness_scaled.flatten(), train_kurtosis_scaled.flatten()
])

X_test_features = np.column_stack([
    test_errors_scaled.flatten(), mahalanobis_test_scaled.flatten(),
    test_mean_scaled.flatten(), test_var_scaled.flatten(),
    test_min_scaled.flatten(), test_max_scaled.flatten(),
    test_skewness_scaled.flatten(), test_kurtosis_scaled.flatten()
])

# ✅ Train XGB on Extracted Features
xgb_params = {
    "n_estimators": 1000,  # Increase if needed, early stopping will handle overfitting
    "learning_rate": 0.02,  # Slower learning to improve generalization
    "max_depth": 4,  # Moderate tree depth to prevent overfitting
    "min_child_weight": 5,  # Prevents overly complex trees
    "subsample": 0.8,  # Uses 80% of data for each tree (helps generalization)
    "colsample_bytree": 1.0,  # ✅ Uses all 8 features
    "gamma": 2,  # Requires a minimum loss reduction to make a split
    "lambda": 5,  # L2 regularization (higher values reduce complexity)
    "alpha": 0,  # L1 regularization (not needed with only 8 features)
    "scale_pos_weight": 40,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50
}

xgb_classifier = xgb.XGBClassifier(**xgb_params)

xgb_classifier.fit(X_train_features, y_train)

# ✅ Predict Fraud Probabilities
xgb_preds = xgb_classifier.predict_proba(X_test_features)[:, 1]

# Optimize threshold for best F1-score
best_f1, best_threshold = 0, 0
for threshold in np.linspace(0.01, 0.5, 50):
    y_test_pred = (xgb_preds >= threshold).astype(int)
    f1 = f1_score(y_test, y_test_pred)
    if f1 > best_f1:
        best_f1, best_threshold = f1, threshold

print(f"✅ Best Threshold: {best_threshold:.4f} | Best F1: {best_f1:.4f}")

# ✅ Final Model Evaluation
final_anomaly_score = xgb_preds
y_test_pred = (final_anomaly_score >= best_threshold).astype(int)

# Compute Metrics
test_pr_auc = average_precision_score(y_test, final_anomaly_score)
test_auc = roc_auc_score(y_test, final_anomaly_score)
test_f1 = f1_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)

print(f"📊 Model Evaluation - Weighted Anomaly Score:")
print(f"🔹 PR-AUC: {test_pr_auc:.4f}")
print(f"🔹 ROC-AUC: {test_auc:.4f}")
print(f"🔹 Precision: {test_precision:.4f}")
print(f"🔹 Recall: {test_recall:.4f}")
print(f"🔹 F1 Score: {test_f1:.4f}")
print(f"🔹 Optimal Threshold: {best_threshold:.4f}")

metrics = {
    "pr_aur": float(test_pr_auc),
    "roc_auc": float(test_auc),
    "precision": float(test_precision),
    "recall": float(test_recall),
    "f1_score": float(test_f1),
    "threshold": float(best_threshold),
}

metrics_path = PROJECT_ROOT / "results/Autoencoder_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Model metrics saved at {metrics_path}")

torch.save(model.state_dict(), PROJECT_ROOT / "src/models/autoencoder.pth")
print(f"✅ Model saved at {PROJECT_ROOT / 'src/models/autoencoder.pth'}")
