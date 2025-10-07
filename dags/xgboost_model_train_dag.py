from datetime import datetime

import logging

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "fraud_detection_project",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 7),
    "max_active_runs": 1,
}


def _train_xgboost_model(**context):
    """Trigger the XGBoost training pipeline and surface key metrics to Airflow."""
    try:
        from xgboost_model_train import XGBoostFraudTrainer

        trainer = XGBoostFraudTrainer()
        _, metrics = trainer.train_model()

        logger.info("XGBoost training finished with metrics: %s", metrics)
        return {"status": "success", "metrics": metrics}
    except Exception as exc:
        logger.error("XGBoost training failed: %s", exc, exc_info=True)
        raise AirflowException(f"XGBoost training failed: {exc}")


with DAG(
    dag_id="fraud_detection_xgboost_training",
    default_args=default_args,
    description="Daily XGBoost fraud detection model training",
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["fraud", "xgboost"],
) as dag:
    validate_environment = BashOperator(
        task_id="validate_environment",
        bash_command="""
        echo "Validating runtime environment..." &&
        test -f /app/config.yaml &&
        test -f /app/.env &&
        echo "Environment validation complete."
        """,
    )

    train_model = PythonOperator(
        task_id="train_xgboost_model",
        python_callable=_train_xgboost_model,
        provide_context=True,
    )

    cleanup = BashOperator(
        task_id="cleanup_temp_files",
        bash_command="rm -f /app/tmp/*.pkl",
        trigger_rule="all_done",
    )

    validate_environment >> train_model >> cleanup

    dag.doc_md = """
    ## Fraud Detection XGBoost Training

    - Consumes recent transactions from Redpanda (Kafka)
    - Trains the XGBoost classifier defined in `dags/xgboost_model_train.py`
    - Logs metrics and model artefacts to MLflow
    """
