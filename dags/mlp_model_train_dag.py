########################################################
# developed by Yikai Cai, 2025-11-05
########################################################

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


def _train_mlp_model(**context):
    """Trigger the MLP training pipeline and surface key metrics to Airflow."""
    try:
        from mlp_model_train import MLPFraudTrainer

        trainer = MLPFraudTrainer()
        _, metrics = trainer.train_model()

        logger.info("MLP training finished with metrics: %s", metrics)
        return {"status": "success", "metrics": metrics}
    except Exception as exc:
        logger.error("MLP training failed: %s", exc, exc_info=True)
        raise AirflowException(f"MLP training failed: {exc}")


with DAG(
    dag_id="fraud_detection_mlp_training",
    default_args=default_args,
    description="Daily MLP fraud detection model training",
    schedule_interval="0 5 * * *",
    catchup=False,
    tags=["fraud", "mlp"],
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
        task_id="train_mlp_model",
        python_callable=_train_mlp_model,
        provide_context=True,
    )

    cleanup = BashOperator(
        task_id="cleanup_temp_files",
        bash_command="rm -f /app/tmp/*.pkl",
        trigger_rule="all_done",
    )

    validate_environment >> train_model >> cleanup

    dag.doc_md = """
    ## Fraud Detection MLP Training

    - Consumes recent transactions from Redpanda (Kafka)
    - Trains the MLP classifier defined in `dags/mlp_model_train.py`
    - Logs metrics and model artefacts to MLflow
    """
