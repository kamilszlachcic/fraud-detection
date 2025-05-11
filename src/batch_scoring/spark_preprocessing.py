import pickle
import os
import pandas as pd


# === STEP 1: Spark-based preprocessing ===

def drop_v_columns_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Drop all columns starting with 'V'."""
    v_cols = [c for c in df.columns if c.startswith('V')]
    return df.drop(*v_cols)


def handle_missing_values_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Basic missing value handling."""
    categorical_cols = [c for (c, dtype) in df.dtypes if dtype == 'string']
    df = df.fillna('Unknown', subset=categorical_cols)

    numeric_cols = [c for (c, dtype) in df.dtypes if dtype in ['double', 'int', 'float', 'bigint']]
    df = df.fillna(0, subset=numeric_cols)

    return df


def add_uid_features_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Add UID feature: combination of card1, card2, addr1."""
    if all(col_name in df.columns for col_name in ["card1", "card2", "addr1"]):
        df = df.withColumn("UID", concat_ws("_", col("card1").cast("string"), col("card2").cast("string"),
                                            col("addr1").cast("string")))
    return df


def add_aggregated_features_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Add aggregated statistical features per UID group."""
    if "UID" not in df.columns:
        return df

    agg_df = df.groupBy("UID").agg(
        F.mean("TransactionAmt").alias("UID_TransactionAmt_mean"),
        F.stddev("TransactionAmt").alias("UID_TransactionAmt_std"),
        F.min("TransactionAmt").alias("UID_TransactionAmt_min"),
        F.max("TransactionAmt").alias("UID_TransactionAmt_max"),
        F.sum("TransactionAmt").alias("UID_TransactionAmt_sum"),
        F.mean("dist1").alias("UID_dist1_mean"),
        F.stddev("dist1").alias("UID_dist1_std"),
        F.min("dist1").alias("UID_dist1_min"),
        F.max("dist1").alias("UID_dist1_max"),
        F.count("TransactionDT").alias("UID_TransactionDT_count")
    )

    df = df.join(agg_df, on="UID", how="left")
    return df


def add_derived_features_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Add simple derived ratio features."""
    if "TransactionAmt" in df.columns and "C1" in df.columns:
        df = df.withColumn("Amt_C1_Ratio", col("TransactionAmt") / (col("C1") + 1))
    if "TransactionAmt" in df.columns and "D1" in df.columns:
        df = df.withColumn("Amt_D1_Ratio", col("TransactionAmt") / (col("D1") + 1))
    if "TransactionAmt" in df.columns and "dist1" in df.columns:
        df = df.withColumn("Amt_Dist1_Ratio", col("TransactionAmt") / (col("dist1") + 1))
    if "TransactionAmt" in df.columns and "TransactionDT" in df.columns:
        df = df.withColumn("Amt_Time_Ratio", col("TransactionAmt") / (col("TransactionDT") + 1))

    return df


def add_frequency_features_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Add frequency encoding feature: count of card1 values."""
    if "card1" not in df.columns:
        return df

    freq_df = df.groupBy("card1").agg(F.count("*").alias("card1_counts"))
    df = df.join(freq_df, on="card1", how="left")

    return df


def load_scaler_bundle():
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Load scaler mean and std dictionaries."""
    scaler_path = os.path.join(MODELS_DIR, "scaler_bundle.pkl")
    with open(scaler_path, "rb") as f:
        scaler_bundle = pickle.load(f)
    return scaler_bundle


def scale_features_spark(df, scaler_bundle):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """Apply scaling using loaded mean and std values."""
    for feature, mean_value in scaler_bundle['mean'].items():
        std_value = scaler_bundle['std'].get(feature, 1.0)
        if feature in df.columns:
            df = df.withColumn(
                feature,
                (col(feature) - float(mean_value)) / float(std_value)
            )
    return df


# === STEP 2: Pandas-based magic features ===

def apply_magic_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply magic features on Pandas DataFrame.
    Note: ONLY call this after converting Spark DataFrame to Pandas.
    """
    df = df.copy()

    if "card1" in df.columns:
        time_diff_card1 = df.groupby("card1")["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card1 = pd.Series(999999, index=df.index)

    if "card2" in df.columns:
        time_diff_card2 = df.groupby("card2")["TransactionDT"].diff().fillna(999999)
    else:
        time_diff_card2 = pd.Series(999999, index=df.index)

    if all(col in df.columns for col in ["card1", "card2", "addr1"]):
        time_diff_card_addr = df.groupby(["card1", "card2", "addr1"])["TransactionDT"].diff().fillna(999999)
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


# === STEP 3: Full preprocessing pipeline ===

def preprocess_and_engineer_features_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, concat_ws
    from src.config import MODELS_DIR

    """
    Full preprocessing pipeline:
    1. Spark transformations
    2. Convert to Pandas
    3. Pandas magic features
    """
    df = handle_missing_values_spark(df)
    df = drop_v_columns_spark(df)
    df = add_uid_features_spark(df)
    df = add_aggregated_features_spark(df)
    df = add_derived_features_spark(df)
    df = add_frequency_features_spark(df)
    df = scale_features_spark(df, load_scaler_bundle())

    # Convert to Pandas
    df = df.toPandas()

    # Apply magic features (Pandas)
    df = apply_magic_features_pandas(df)

    # Drop UID
    df.drop(columns=["UID"], inplace=True, errors="ignore")

    print(f"✅ Full preprocessing complete! Final shape: {df.shape}")
    return df