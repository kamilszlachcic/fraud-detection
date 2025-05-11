import os
import pickle
from src.config import SPLIT_DATA_DIR, BATCH_PREDICTIONS_DIR, XGB_MODEL_PATH
from src.batch_scoring.spark_preprocessing import preprocess_and_engineer_features_spark
import pandas as pd


def load_model_bundle():
    """
    Initialize SparkSession and load the trained XGBoost model bundle.
    Returns:
        tuple: (SparkSession object, trained XGBoost model, feature names)
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("FraudDetectionBatchScoring") \
        .getOrCreate()

    with open(XGB_MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    return spark, model, feature_names


def list_chunk_names():
    """
    List all available chunk names based on split data files.
    """
    files = os.listdir(SPLIT_DATA_DIR)
    chunk_names = sorted(
        list(set(f.split("_")[0] for f in files if f.endswith("_transaction.csv")))
    )
    return chunk_names


def load_chunk(spark, chunk_name):
    """
    Load transaction and identity split files into Spark DataFrames.
    """
    transaction_path = os.path.join(SPLIT_DATA_DIR, f"{chunk_name}_transaction.csv")
    identity_path = os.path.join(SPLIT_DATA_DIR, f"{chunk_name}_identity.csv")

    transaction_df = spark.read.csv(transaction_path, header=True, inferSchema=True)
    identity_df = spark.read.csv(identity_path, header=True, inferSchema=True)

    return transaction_df, identity_df


def preprocess_and_predict_chunk(transaction_df, identity_df, model, feature_names):

    """
    Full batch scoring pipeline: Spark preprocessing + Pandas feature engineering + model prediction.
    """
    from pyspark.sql import SparkSession

    # Merge transaction and identity data
    df = transaction_df.join(identity_df, on='TransactionID', how='left')

    # Full preprocessing: Spark + Pandas (hybrid)
    df = preprocess_and_engineer_features_spark(df)

    # Ensure correct columns are selected
    df = df.set_index("TransactionID", drop=False)
    df_selected = df[feature_names]

    # Predict fraud probability
    probs = model.predict_proba(df_selected)[:, 1]

    # Prepare output DataFrame
    output = pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "fraud_probability": probs
    })

    return output


def save_predictions(predictions_df, chunk_name):
    """
    Save predictions to CSV.
    """
    output_path = os.path.join(BATCH_PREDICTIONS_DIR, f"{chunk_name}_predictions.csv")
    predictions_df.to_csv(output_path, index=False)
