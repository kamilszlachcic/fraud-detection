import pandas as pd

# === STEP 1: Handle Missing Values ===
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill object columns with "Unknown"
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Fill numeric-like categorical features
    fraud_sensitive_features = ["TransactionAmt"] + [col for col in df.columns if col.startswith('V')]
    categorical_like_numeric = [col for col in df.columns if col.startswith(('C', 'D', 'dist'))]
    df[categorical_like_numeric] = df[categorical_like_numeric].fillna(0)
    df[fraud_sensitive_features] = df[fraud_sensitive_features].apply(lambda x: x.fillna(x.median()))

    # Fill card/email-related with mode
    card_email_features = ["card4", "card6", "P_emaildomain", "R_emaildomain"]
    for col in card_email_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# === STEP 2: Encode categoricals ===
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_enc_cols = [
        "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
        "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
        "DeviceType", "DeviceInfo", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"
    ]

    for col in categorical_enc_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df


# === STEP 3: UID feature ===
def add_uid_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["UID"] = df[["card1", "card2", "addr1"]].astype(str).agg("_".join, axis=1)
    return df


# === STEP 4: Aggregated features ===
def add_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("UID").agg({
        "TransactionAmt": ["mean", "std", "min", "max", "sum"],
        "dist1": ["mean", "std", "min", "max"],
        "TransactionDT": ["count"],
    }).reset_index()

    agg.columns = ["UID"] + [f"UID_" + "_".join(col) for col in agg.columns[1:]]
    df = pd.merge(df, agg, on="UID", how="left", copy=False)
    df.fillna(0, inplace=True)

    return df


# === STEP 5: Derived features ===
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amt_C1_Ratio"] = df["TransactionAmt"] / (df["C1"] + 1)
    df["Amt_D1_Ratio"] = df["TransactionAmt"] / (df["D1"] + 1)
    df["Amt_Dist1_Ratio"] = df["TransactionAmt"] / (df["dist1"] + 1)
    df["Amt_Time_Ratio"] = df["TransactionAmt"] / (df["TransactionDT"] + 1)
    return df


# === STEP 6: Frequency encoding ===
def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["card1_counts"] = df["card1"].map(df["card1"].value_counts())
    return df


# === STEP 7: Magic features ===
def apply_magic_features(df: pd.DataFrame, group_cols=["card1", "card2", "addr1"]) -> pd.DataFrame:
    df = df.copy()

    if "card1" in df.columns:
        time_diff_card1 = df.groupby("card1")["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card1 = pd.Series(999999, index=df.index)

    if "card2" in df.columns:
        time_diff_card2 = df.groupby("card2")["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card2 = pd.Series(999999, index=df.index)

    if all(col in df.columns for col in group_cols):
        time_diff_card_addr = df.groupby(group_cols)["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card_addr = pd.Series(999999, index=df.index)

    freq_cols = ["card1", "card2", "addr1"]
    count_features = {col: df[col].map(df[col].value_counts()) for col in freq_cols}
    mean_transaction_amt = {col: df.groupby(col)["TransactionAmt"].transform("mean") for col in freq_cols}
    transaction_amt_ratio = {
        col: df["TransactionAmt"] / (mean_transaction_amt[col] + 1e-6) for col in freq_cols
    }

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


# === FINAL: Full preprocessing pipeline ===
def preprocess_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    df = add_uid_features(df)
    df = add_aggregated_features(df)
    df = add_derived_features(df)
    df = add_frequency_features(df)
    df = apply_magic_features(df)
    df = df.drop(columns=["UID"], errors="ignore")

    print(f"✅ Preprocessing complete! Final shape: {df.shape}")
    return df
