# Environment Setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# Dataset load
dataset = pd.read_csv('dataset/train_transaction.csv')
dataset.set_index("TransactionID", inplace=True)
print(dataset.info())
print(dataset.describe())

# Thanks to spectacular EDA for Columns V https://www.kaggle.com/code/cdeotte/eda-for-columns-v-and-id#V-Reduced
# 211 V-Columns will be droped

# Define Selected all V-Columns
all_v_columns = [col for col in dataset.columns if col.startswith("V")]

# Define Selected V-Columns
v = [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]
v += [96, 98, 99, 104]
v += [107, 108, 111, 115, 117, 120, 121, 123]
v += [124, 127, 129, 130, 136]
v += [138, 139, 142, 147, 156, 162]
v += [165, 160, 166]
v += [178, 176, 173, 182]
v += [187, 203, 205, 207, 215]
v += [169, 171, 175, 180, 185, 188, 198, 210, 209]
v += [218, 223, 224, 226, 228, 229, 235]
v += [240, 258, 257, 253, 252, 260, 261]
v += [264, 266, 267, 274, 277]
v += [220, 221, 234, 238, 250, 271]
v += [294, 284, 285, 286, 291, 297]
v += [303, 305, 307, 309, 310, 320]
v += [281, 283, 289, 296, 301, 314]
v += [332, 325, 335, 338]

# Convert to column names
selected_v_columns = [f"V{i}" for i in v]

# Determine unimportant V-columns (all V-columns - selected V-columns)
not_important_v_columns = list(set(all_v_columns) - set(selected_v_columns))

# Drop unimportant V-columns while keeping everything else
dataset_filtered = dataset.drop(columns=not_important_v_columns)

# Print the shape to confirm changes
print(f"Original Dataset Shape: {dataset.shape}")
print(f"Filtered Dataset Shape: {dataset_filtered.shape}")
print(f"Dropped {len(not_important_v_columns)} Unimportant V-Columns")

# Convert D Columns into Actual Past Time Points
for i in range(1, 16):  # D1 to D15
    if i in [1, 2, 3, 5, 9]: continue  # Skip these columns
    if f'D{i}' in dataset_filtered.columns:
        dataset_filtered[f'D{i}'] = dataset_filtered[f'D{i}'] - dataset_filtered['TransactionDT'] / np.float32(
            24 * 60 * 60)

# Standardize TransactionAmt (If Available)
if 'TransactionAmt' in dataset_filtered.columns:
    scaler = StandardScaler()
    dataset_filtered['TransactionAmt_scaled'] = scaler.fit_transform(dataset_filtered[['TransactionAmt']])

# Feature Engineering for Time-Based Features
dataset_filtered['TransactionDay'] = dataset_filtered['TransactionDT'] // (24 * 3600)
dataset_filtered['TransactionHour'] = (dataset_filtered['TransactionDT'] % (24 * 3600)) // 3600

# Feature Engineering for categorical Freatures
categorical_cols = dataset_filtered.select_dtypes(include=['object']).columns
print("🔹 Categorical Columns:", categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    dataset_filtered[col] = le.fit_transform(dataset_filtered[col].astype(str))

print(dataset_filtered.dtypes.value_counts())

# Final Dataset After Feature Engineering
X = dataset_filtered.drop(columns=['isFraud'])
y = dataset_filtered['isFraud']

# Fill Missing Values
X = X.fillna(0)

print(f"Feature Engineering Complete! Final Dataset Shape: {X.shape}")

# Stratified Split (95% Train / 5% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=42
)

print(f"Training Size: {X_train.shape[0]} rows")
print(f"Test Size: {X_test.shape[0]} rows")

fraud_count = y_train.value_counts()
print(f"Non-Fraud Cases in Training: {fraud_count[0]}")
print(f"Fraud Cases in Training: {fraud_count[1]}")
print(f"Fraud Ratio in Training: {fraud_count[1] / (fraud_count[0] + fraud_count[1]) * 100:.4f}%")

print("\n🔹 Running XGBoost for Feature Importance Analysis...")

# Feature Selection Using XGBoost

# Initialize XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    missing=-1,
    eval_metric='auc',
    nthread=4,
    tree_method='hist'
)

# Train XGBoost on Full Feature Set
xgb_model.fit(X_train, y_train, verbose=30)

# Get Feature Importances
importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Select Top Features Based on Importance
N_FEATURES = 100
top_features = feature_importance_df.iloc[:N_FEATURES]["Feature"].tolist()
# Print Feature Importances
print("\n🔹 Top 10 Features Based on XGBoost Importance:")
print(top_features[:10])


# Create New Training Data with Only These Features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

print(f"Training Data Shape After Feature Selection: {X_train_selected.shape}")

smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_selected, y_train = smote.fit_resample(X_train_selected, y_train)


X_train_tensor = torch.tensor(X_train_selected.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_selected.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# Define & Train the Model
class FraudDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FraudDetectionNN(input_dim=X_train_tensor.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00005)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.85, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

criterion = FocalLoss()

model_save_path = "Fraud_Detection/src/models/NN_model.pth"
best_auc = 0.0  # Track the best ROC-AUC
num_epochs = 75
patience = 5
counter = 0
threshold = 0.3

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    y_train_true = []
    y_train_pred = []

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Store predictions & true values for accuracy calculation
        y_train_true.extend(batch_y.cpu().numpy())
        y_train_pred.extend(outputs.cpu().detach().numpy())

    epoch_loss /= len(train_loader)

    # Convert predictions to binary
    y_train_pred = torch.tensor(y_train_pred)
    y_train_pred_class = (y_train_pred > threshold).int()

    # Compute Training Metrics
    train_accuracy = accuracy_score(y_train_true, y_train_pred_class)
    train_precision = precision_score(y_train_true, y_train_pred_class, zero_division=0)
    train_recall = recall_score(y_train_true, y_train_pred_class, zero_division=0)
    train_f1 = f1_score(y_train_true, y_train_pred_class, zero_division=0)
    train_auc = roc_auc_score(y_train_true, y_train_pred)

    # Check Model Predictions
    print("Sample Predictions (First 10):", outputs[:10].cpu().detach().numpy())
    print("Min Prediction Value:", np.min(outputs.cpu().detach().numpy()))
    print("Max Prediction Value:", np.max(outputs.cpu().detach().numpy()))

    # Print Training Metrics After Each Epoch
    print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.8f} | Acc: {train_accuracy:.4f} | "
          f"Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | "
          f"F1: {train_f1:.4f} | AUC: {train_auc:.4f}")

    # Save the best model based on ROC-AUC
    if train_auc > best_auc:
        best_auc = train_auc
        torch.save(model.state_dict(), model_save_path)
        counter = 0
    else:
        counter += 1

    # Early Stopping Condition
    if counter >= patience:
        print(f"⏹️ Early stopping triggered at Epoch {epoch + 1} (No improvement for {patience} epochs).")
        break

print(f"Best model saved at Epoch {epoch + 1} with AUC: {best_auc:.4f}")
print("\nModel Training Complete!")

print("\n🔹 Loading Best Model for Final Evaluation...")

# Load the best saved model
model.load_state_dict(torch.load(model_save_path))
model.eval()

y_test_pred = []
y_test_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        y_test_pred.extend(outputs.cpu().numpy())
        y_test_true.extend(batch_y.cpu().numpy())

# Convert predictions to binary
y_test_pred = torch.tensor(y_test_pred)
y_test_pred_class = (y_test_pred > threshold).int()

# Compute Test Metrics
test_accuracy = accuracy_score(y_test_true, y_test_pred_class)
test_precision = precision_score(y_test_true, y_test_pred_class, zero_division=0)
test_recall = recall_score(y_test_true, y_test_pred_class, zero_division=0)
test_f1 = f1_score(y_test_true, y_test_pred_class, zero_division=0)
test_auc = roc_auc_score(y_test_true, y_test_pred)

# Print Final Test Results
print("\nFinal Test Set Metrics (Best Model):")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print(f"ROC-AUC: {test_auc:.4f}")
