import logging
from typing import Dict, List

import mlflow

from logistic_model_train import LogisticFraudTrainer


logger = logging.getLogger(__name__)

CANDIDATE_PARAMS: List[Dict[str, object]] = [
    {"solver": "lbfgs", "penalty": "l2", "C": 0.5, "class_weight": "balanced"},
    {"solver": "lbfgs", "penalty": "l2", "C": 0.2, "class_weight": {0: 1, 1: 120}},
    {"solver": "lbfgs", "penalty": "l2", "C": 0.05, "class_weight": {0: 1, 1: 200}},
    {"solver": "saga", "penalty": "l1", "C": 0.1, "class_weight": "balanced", "max_iter": 3000},
    {
        "solver": "saga",
        "penalty": "elasticnet",
        "C": 0.2,
        "l1_ratio": 0.5,
        "class_weight": "balanced",
        "max_iter": 3000,
    },
    {"solver": "saga", "penalty": "l2", "C": 1.5, "class_weight": "balanced", "max_iter": 3000},
]


def run_search() -> None:
    trainer = LogisticFraudTrainer()
    df = trainer.read_from_kafka()
    logger.info("Loaded %s rows for logistic regression hyperparameter search.", len(df))

    results = []

    with mlflow.start_run(run_name="logistic_hp_search"):
        for idx, params in enumerate(CANDIDATE_PARAMS, start=1):
            combo_label = f"logreg_combo_{idx:02d}"
            logger.info("Training %s with params: %s", combo_label, params)
            _, metrics = trainer.train_model(
                df=df,
                params_override=params,
                run_name=combo_label,
                log_to_mlflow=True,
                persist_model=False,
            )
            results.append({"label": combo_label, "params": params, "metrics": metrics})
            mlflow.log_metric(f"{combo_label}_avg_precision", metrics["avg_precision"])

        if not results:
            logger.warning("No logistic hyperparameter candidates evaluated.")
            return

        best_result = max(results, key=lambda item: item["metrics"]["avg_precision"])
        logger.info(
            "Best combo %s: avg_precision=%.4f precision=%.4f recall=%.4f params=%s",
            best_result["label"],
            best_result["metrics"]["avg_precision"],
            best_result["metrics"]["precision"],
            best_result["metrics"]["recall"],
            best_result["params"],
        )
        for key, value in best_result["params"].items():
            mlflow.log_param(f"best_{key}", value)
        mlflow.log_metric("best_avg_precision", best_result["metrics"]["avg_precision"])
        mlflow.log_metric("best_precision", best_result["metrics"]["precision"])
        mlflow.log_metric("best_recall", best_result["metrics"]["recall"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_search()
