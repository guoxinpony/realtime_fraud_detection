import logging
from itertools import product
from typing import Dict, List

import mlflow

from xgboost_model_train import XGBoostFraudTrainer


logger = logging.getLogger(__name__)

HYPERPARAM_GRID: Dict[str, List[float]] = {
    "max_depth": [4, 6],
    "min_child_weight": [1, 4],
    "gamma": [0.0, 0.5],
    "subsample": [0.85, 0.95],
    "colsample_bytree": [0.8, 0.95],
    "learning_rate": [0.05, 0.08],
    "n_estimators": [120, 200],
    "max_delta_step": [0, 1],
}

MAX_COMBINATIONS = 10


def _candidate_params() -> List[Dict[str, float]]:
    keys = list(HYPERPARAM_GRID.keys())
    values = [HYPERPARAM_GRID[key] for key in keys]
    combos: List[Dict[str, float]] = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
        if len(combos) >= MAX_COMBINATIONS:
            break
    return combos


def run_search() -> None:
    trainer = XGBoostFraudTrainer()
    df = trainer.read_from_kafka()
    total_rows = len(df)
    logger.info("Starting XGBoost hyperparameter search on %d rows.", total_rows)

    candidates = _candidate_params()
    logger.info("Evaluating %d candidate settings (limited for runtime safety).", len(candidates))

    results = []

    with mlflow.start_run(run_name="xgboost_hp_search"):
        for idx, params in enumerate(candidates, start=1):
            combo_label = f"hp_combo_{idx:02d}"
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
            logger.warning("No hyperparameter candidates were evaluated.")
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
