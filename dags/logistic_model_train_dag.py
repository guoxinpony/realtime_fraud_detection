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


def _train_logistic_model(**context):
    """Trigger the logistic regression training pipeline and surface key metrics to Airflow."""
    try:
        from logistic_model_train import LogisticFraudTrainer

        trainer = LogisticFraudTrainer()
        _, metrics = trainer.train_model()

        logger.info("Logistic regression training finished with metrics: %s", metrics)
        return {"status": "success", "metrics": metrics}
    except Exception as exc:
        logger.error("Logistic regression training failed: %s", exc, exc_info=True)
        raise AirflowException(f"Logistic regression training failed: {exc}")


with DAG(
    dag_id="fraud_detection_logistic_training",
    default_args=default_args,
    description="Daily logistic regression fraud detection model training",
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["fraud", "logistic_regression"],
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
        task_id="train_logistic_model",
        python_callable=_train_logistic_model,
        provide_context=True,
    )

    cleanup = BashOperator(
        task_id="cleanup_temp_files",
        bash_command="rm -f /app/tmp/*.pkl",
        trigger_rule="all_done",
    )

    validate_environment >> train_model >> cleanup

    dag.doc_md = """
    ## Fraud Detection Logistic Regression Training

    - Consumes recent transactions from Redpanda (Kafka)
    - Trains the logistic regression classifier defined in `dags/logistic_model_train.py`
    - Logs metrics and model artefacts to MLflow
    """
