from pathlib import Path
import joblib
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

df1_path = PROJECT_ROOT / "data/raw/test_transaction.csv"
df2_path = PROJECT_ROOT / "data/raw/test_identity.csv"

df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)

dataset = df1.merge(df2, how='left')
dataset.columns = dataset.columns.str.replace("-", "_")
print(dataset.info())
dataset.set_index("TransactionID", inplace=True)

# Thanks to spectacular EDA for Columns V https://www.kaggle.com/code/cdeotte/eda-for-columns-v-and-id#V-Reduced
# 211 V-Columns will be droped
# Define Selected all V-Columns
all_v_columns = [col for col in dataset.columns if col.startswith("V")]
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

def apply_magic_features(df, group_cols=["card1", "card2", "addr1"]):
    """
    Applies advanced feature engineering based on Chris Deotte's 'Magic Features' approach.

    Parameters:
    - df: The dataset (train, test, or submission)
    - group_cols: Columns used for grouping and extracting time-based patterns

    Returns:
    - df with new "magic" features
    """

    df = df.copy()

    # 1️⃣ **Compute time difference features** (grouped by card)
    if "card1" in df.columns:
        time_diff_card1 = df.groupby("card1")["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card1 = pd.Series(999999, index=df.index)

    if "card2" in df.columns:
        time_diff_card2 = df.groupby("card2")["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card2 = pd.Series(999999, index=df.index)

    group_cols = ["card1", "card2", "addr1"]
    if all(col in df.columns for col in group_cols):
        time_diff_card_addr = df.groupby(group_cols)["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card_addr = pd.Series(999999, index=df.index)

    # 2️⃣ **Frequency Encoding (Count Features)**
    freq_cols = ["card1", "card2", "addr1"]
    count_features = {col: df[col].map(df[col].value_counts()) for col in freq_cols}

    # 3️⃣ **Mean Transaction Amount Per Group**
    mean_transaction_amt = {col: df.groupby(col)["TransactionAmt"].transform("mean") for col in freq_cols}

    # 4️⃣ **Transaction Amount Ratio (Normalized Spending)**
    transaction_amt_ratio = {
        col: df["TransactionAmt"] / (mean_transaction_amt[col] + 1e-6) for col in freq_cols
    }

    # ✅ **Batch update DataFrame to avoid fragmentation**
    new_features = pd.DataFrame({
        "time_diff_card1": time_diff_card1,
        "time_diff_card2": time_diff_card2,
        "time_diff_card_addr": time_diff_card_addr,
        **{f"{col}_count": count_features[col] for col in freq_cols},
        **{f"{col}_TransactionAmt_mean": mean_transaction_amt[col] for col in freq_cols},
        **{f"{col}_TransactionAmt_ratio": transaction_amt_ratio[col] for col in freq_cols},
    })

    df = pd.concat([df, new_features], axis=1)

    print(f"✅ Applied magic features! New shape: {df.shape}")
    return df


def preprocess_and_engineer_features(df, test_size=0.2, random_state=42):
    """Preprocess data: handle missing values, encoding, feature engineering, scaling, and train-test split."""

    df = df.copy()  # Prevent modifying original data

    # 1️⃣ **Handle Missing Values**
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(exclude=["object"]).columns

    # Fill categorical missing values with "Unknown"
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Define feature groups
    fraud_sensitive_features = ["TransactionAmt"] + [col for col in df.columns if col.startswith('V')]
    categorical_like_numeric = [col for col in df.columns if col.startswith(('C', 'D', 'dist'))]

    # Fill missing values
    df[categorical_like_numeric] = df[categorical_like_numeric].fillna(0)  # Fill categorical-like numeric with 0
    df[fraud_sensitive_features] = df[fraud_sensitive_features].apply(
        lambda x: x.fillna(x.median()))  # Fill with median

    # Fill missing card & email-related features with mode
    card_email_features = ["card4", "card6", "P_emaildomain", "R_emaildomain"]
    for col in card_email_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 2️⃣ **Categorical Encoding**
    def encode_categorical_features(df, categorical_cols):
        """Encodes categorical columns safely, handling unseen categories."""
        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes  # ✅ Assigns -1 to unseen categories
        return df

    categorical_enc_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
                            "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
                            "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
                            "DeviceType", "DeviceInfo", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

    df = encode_categorical_features(df, categorical_enc_cols)

    # 3️⃣ **Fix Performance Warning - Create UID Efficiently**
    df = df.copy()  # Defragmentation Trick!
    df["UID"] = df[["card1", "card2", "addr1"]].astype(str).agg("_".join, axis=1)

    # 4️⃣ **Feature Aggregation (Grouped Statistics)**
    agg_features = df.groupby("UID").agg({
        "TransactionAmt": ["mean", "std", "min", "max", "sum"],  # Spending behavior
        "dist1": ["mean", "std", "min", "max"],  # Distance-based anomalies
        "TransactionDT": ["count"],  # Transaction frequency
    }).reset_index()

    # Rename columns
    agg_features.columns = ["UID"] + [f"UID_" + "_".join(col) for col in agg_features.columns[1:]]

    # Merge aggregated features (Use `merge()` efficiently)
    df = pd.merge(df, agg_features, on="UID", how="left", copy=False)

    # Fill missing values in aggregation columns
    df.fillna(0, inplace=True)

    # 5️⃣ **Derived Features (New Features)**
    df["Amt_C1_Ratio"] = df["TransactionAmt"] / (df["C1"] + 1)  # Prevent division by zero
    df["Amt_D1_Ratio"] = df["TransactionAmt"] / (df["D1"] + 1)
    df["Amt_Dist1_Ratio"] = df["TransactionAmt"] / (df["dist1"] + 1)
    df["Amt_Time_Ratio"] = df["TransactionAmt"] / (df["TransactionDT"] + 1)

    # 6️⃣ **Frequency Encoding**
    df["card1_counts"] = df["card1"].map(df["card1"].value_counts())

    # 7️⃣ **Drop unused columns**
    df.drop(columns=["isFraud", "UID"], errors="ignore")

    # 8️⃣ **Feature Scaling**
    columns_to_scale = joblib.load(PROJECT_ROOT / "data/processed/scaled_columns.pkl")
    scaler = joblib.load(PROJECT_ROOT / "data/processed/standard_scaler.pkl")
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    print(f"✅ Preprocessing Complete! Dataset Shape: {df.shape}")
    return df


dataset_handled = apply_magic_features(dataset_filtered)
dataset_handled = preprocess_and_engineer_features(dataset_handled)

feature_names = joblib.load(PROJECT_ROOT / "data/processed/feature_names.pkl")
dataset_handled = dataset_handled.reindex(columns=feature_names, fill_value=0)


non_numeric_columns = dataset_handled.select_dtypes(exclude=['number']).columns
print(f"⚠️ Non-numeric columns found in Dataset: {list(non_numeric_columns)}")

missing_per_column = dataset_handled.isnull().sum()
print(missing_per_column[missing_per_column > 0])
print(dataset_handled.dtypes.value_counts())

dataset_handled = dataset_handled.astype({col: "int32" for col in dataset_handled.select_dtypes(include=["int64"]).columns})
dataset_handled = dataset_handled.astype({col: "float32" for col in dataset_handled.select_dtypes(include=["float64"]).columns})
print("✅ Converted all int64 columns to int32 and all float64 to float32")

print(dataset_handled.dtypes.value_counts())

processed_data_path = PROJECT_ROOT / "src" / "sample_data" / "inference_input_processed.pkl"
processed_data_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"X_input": dataset_handled}, processed_data_path)
