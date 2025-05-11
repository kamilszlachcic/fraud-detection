import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import pendulum

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=5),
}

@dag(
    dag_id='batch_scoring_pipeline',
    description='Batch scoring pipeline using PySpark preprocessing and XGBoost model.',
    schedule=None,
    start_date=pendulum.now().subtract(days=1),
    catchup=False,
    tags=['batch-scoring', 'pyspark', 'xgboost', 'fraud-detection'],
)
def batch_scoring_pipeline():

    @task
    def initialize_model():
        from src.batch_scoring.batch_scorer import load_model_bundle  # <<< import lokalnie w tasku!
        spark, model, feature_names = load_model_bundle()
        return {"spark_session_id": id(spark), "model": model, "feature_names": feature_names}

    @task
    def score_chunks(context):
        from src.batch_scoring.batch_scorer import (
            list_chunk_names,
            load_chunk,
            preprocess_and_predict_chunk,
            save_predictions
        )
        from pyspark.sql import SparkSession  # też lokalny import, just in case

        spark = SparkSession.builder.getOrCreate()

        model = context["model"]
        feature_names = context["feature_names"]

        chunk_names = list_chunk_names()

        for chunk_name in chunk_names:
            transaction_df, identity_df = load_chunk(spark, chunk_name)
            predictions_df = preprocess_and_predict_chunk(transaction_df, identity_df, model, feature_names)
            save_predictions(predictions_df, chunk_name)

        return True

    model_context = initialize_model()
    score_chunks(model_context)


dag_instance = batch_scoring_pipeline()
