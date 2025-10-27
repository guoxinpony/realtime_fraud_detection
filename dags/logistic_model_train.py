import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from kafka import KafkaConsumer
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LogisticFraudTrainer:
    """Train a logistic regression classifier for fraud detection using Kafka sourced data."""

    def __init__(self, config_path: Optional[str] = None, sample_size: Optional[int] = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = Path(config_path or "/app/config.yaml")
        if not self.config_path.exists():
            self.config_path = self.project_root / "config.yaml"

        self._load_env_files()
        self.config = self._load_config(self.config_path)

        self.kafka_config: Dict[str, object] = self.config.get("kafka", {})
        self.data_config: Dict[str, object] = self.config.get("data", {})

        models_config = self.config.get("models", {})
        self.model_config: Dict[str, object] = models_config.get("logistic", {})
        if not self.model_config:
            raise ValueError("Configuration is missing `models.logistic` settings.")

        self.registered_model_name = self.model_config.get(
            "mlflow_registered_model_name",
            self.config.get("mlflow", {}).get("registered_model_name"),
        )

        self.max_messages = sample_size or int(self.kafka_config.get("max_messages", 20000))

        os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", str(self.kafka_config.get("bootstrap_servers", "")))

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", self.config["mlflow"]["s3_endpoint_url"])

    def _load_env_files(self) -> None:
        """Load environment variables from known locations."""
        for candidate in (Path("/app/.env"), self.project_root / ".env"):
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=False)

    @staticmethod
    def _load_config(config_path: Path) -> Dict[str, object]:
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        except Exception as exc:
            logger.error("Unable to load configuration from %s: %s", config_path, exc)
            raise

    def _resolve_path(self, path_str: str) -> Path:
        target = Path(path_str)
        if target.is_absolute():
            return target
        return (self.project_root / path_str).resolve()

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map different payload schemas to a unified set of feature columns."""
        df = df.copy()
        column_aliases = {
            "transacted_at": "transaction_time",
            "trans_date_trans_time": "transaction_time",
            "amt": "amount",
            "amount": "amount",
            "merch_lat": "merchant_lat",
            "merchant_lat": "merchant_lat",
            "merch_long": "merchant_long",
            "merchant_long": "merchant_long",
            "cc_num": "card_number",
            "trans_num": "transaction_id",
        }

        for source, target in column_aliases.items():
            if source in df.columns and target not in df.columns:
                df[target] = df[source]

        required = ["transaction_time", "amount", "merchant", "category", "city", "state", "zip", "lat", "long",
                    "city_pop", "merchant_lat", "merchant_long", "is_fraud"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce", utc=True)
        for column in ["amount", "zip", "lat", "long", "city_pop", "merchant_lat", "merchant_long"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["transaction_time", "amount", "is_fraud"])
        df["is_fraud"] = df["is_fraud"].astype(int)

        return df

    def _load_csv_fallback(self) -> pd.DataFrame:
        csv_path = self.data_config.get("training_csv")
        if not csv_path:
            raise ValueError("Configuration is missing `data.training_csv`.")
        path = self._resolve_path(str(csv_path))
        if not path.exists():
            raise FileNotFoundError(f"Training CSV not found: {path}")

        logger.warning("Kafka returned no data; falling back to CSV at %s", path)
        df = pd.read_csv(path)
        df = self._normalise_columns(df)
        df = df.sort_values("transaction_time").tail(self.max_messages)
        return df

    def _build_consumer(self) -> KafkaConsumer:
        security_protocol = str(self.kafka_config.get("security_protocol", "PLAINTEXT")).upper()
        bootstrap = str(self.kafka_config.get("bootstrap_servers", ""))
        if not bootstrap:
            raise ValueError("Kafka bootstrap servers are not configured.")

        topic = self.kafka_config.get("topic")
        if not topic:
            raise ValueError("Kafka topic is not configured.")

        sasl_kwargs: Dict[str, object] = {}
        if security_protocol.startswith("SASL"):
            sasl_kwargs = {
                "sasl_mechanism": self.kafka_config.get("sasl_mechanism", "PLAIN"),
                "sasl_plain_username": self.kafka_config.get("username"),
                "sasl_plain_password": self.kafka_config.get("password"),
            }

        consumer = KafkaConsumer(
            str(topic),
            bootstrap_servers=[server.strip() for server in bootstrap.split(",") if server.strip()],
            security_protocol=security_protocol,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="earliest",
            consumer_timeout_ms=int(self.kafka_config.get("timeout", 10000)),
            **sasl_kwargs,
        )
        return consumer

    def read_from_kafka(self) -> pd.DataFrame:
        """Read recent messages from Kafka and convert to DataFrame."""
        try:
            consumer = self._build_consumer()
        except Exception:
            logger.exception("Failed to construct Kafka consumer; using CSV fallback.")
            return self._load_csv_fallback()

        logger.info("Consuming messages from Kafka topic %s", self.kafka_config.get("topic"))
        messages = []

        try:
            for message in consumer:
                messages.append(message.value)
                if len(messages) >= self.max_messages:
                    break
        finally:
            consumer.close()

        if not messages:
            logger.warning("No Kafka messages retrieved; using CSV fallback.")
            return self._load_csv_fallback()

        df = pd.DataFrame(messages)
        df = self._normalise_columns(df)
        df = df.sort_values("transaction_time").tail(self.max_messages)
        logger.info("Kafka ingestion completed with %s records.", len(df))
        return df

    @staticmethod
    def _build_feature_pipeline() -> ColumnTransformer:
        numeric_features = [
            "amount",
            "zip",
            "lat",
            "long",
            "city_pop",
            "merchant_lat",
            "merchant_long",
            "transaction_hour",
            "transaction_dayofweek",
            "transaction_month",
            "amount_log",
            "amount_zscore",
            "merchant_distance",
            "city_pop_log",
        ]
        categorical_features = ["category", "state", "gender", "city", "merchant"]

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical", categorical_transformer, categorical_features),
            ]
        )

        return preprocessor

    @staticmethod
    def _augment_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["transaction_hour"] = df["transaction_time"].dt.hour.astype(int)
        df["transaction_dayofweek"] = df["transaction_time"].dt.dayofweek.astype(int)
        df["transaction_month"] = df["transaction_time"].dt.month.astype(int)
        df["gender"] = df.get("gender", "unknown").fillna("unknown")
        df["amount_log"] = np.log1p(df["amount"].clip(lower=0))
        df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-6)
        df["merchant_distance"] = np.sqrt(
            (df["lat"] - df["merchant_lat"]) ** 2 + (df["long"] - df["merchant_long"]) ** 2
        )
        df["city_pop_log"] = np.log1p(df["city_pop"].clip(lower=0))
        return df

    def _rebalance_dataframe(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        ratio = float(self.model_config.get("negative_downsample_ratio", 1.0))
        if ratio >= 1 or ratio <= 0:
            return df

        positives = df[df["is_fraud"] == 1]
        negatives = df[df["is_fraud"] == 0]
        if negatives.empty:
            return df

        keep_negatives = int(max(len(negatives) * ratio, len(positives)))
        keep_negatives = min(keep_negatives, len(negatives))
        if keep_negatives <= 0:
            logger.warning("Downsample ratio %.3f removed all negative samples; skipping downsampling.", ratio)
            return df

        sampled_negatives = negatives.sample(n=keep_negatives, random_state=seed, replace=False)
        balanced = (
            pd.concat([positives, sampled_negatives])
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

        logger.info(
            "Downsampled negatives from %s to %s (ratio=%.2f); resulting rows=%s",
            len(negatives),
            keep_negatives,
            ratio,
            len(balanced),
        )
        return balanced

    def train_model(
        self,
        df: Optional[pd.DataFrame] = None,
        params_override: Optional[Dict[str, float]] = None,
        run_name: Optional[str] = None,
        log_to_mlflow: bool = True,
        persist_model: bool = True,
    ) -> Tuple[Pipeline, Dict[str, float]]:
        logger.info("Starting logistic regression fraud detection training workflow.")
        if df is None:
            df = self.read_from_kafka()
        else:
            df = df.copy()
        df = self._augment_features(df)

        test_size = float(self.model_config.get("test_size", 0.2))
        validation_size = float(self.model_config.get("validation_size", 0.1))
        threshold_recall_target = float(self.model_config.get("threshold_recall_target", 0.9))
        downsample_ratio = float(self.model_config.get("negative_downsample_ratio", 1.0))
        seed = int(self.model_config.get("seed", 42))

        df = self._rebalance_dataframe(df, seed)

        features = df.drop(columns=["is_fraud", "transaction_time"])
        target = df["is_fraud"]

        if target.sum() == 0:
            raise ValueError("No fraud samples present in the training data.")

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            stratify=target,
            random_state=seed,
        )

        X_train, y_train = X_train_full, y_train_full
        X_val = y_val = None
        if 0 < validation_size < 1:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size=validation_size,
                    stratify=y_train_full,
                    random_state=seed,
                )
            except ValueError as exc:
                logger.warning("Skipping validation split (validation_size=%s): %s", validation_size, exc)
                X_train, y_train = X_train_full, y_train_full
                X_val = y_val = None

        preprocessor = self._build_feature_pipeline()

        params = dict(self.model_config.get("params", {}))
        if params_override:
            params.update(params_override)
        params.setdefault("solver", "lbfgs")
        params.setdefault("max_iter", 1000)
        params.setdefault("class_weight", "balanced")

        model = LogisticRegression(
            random_state=seed,
            **params,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        run_ctx = (
            mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
            if log_to_mlflow
            else nullcontext()
        )

        with run_ctx:
            if log_to_mlflow:
                mlflow.log_params(
                    {
                        f"classifier__{k}": json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
                        for k, v in params.items()
                    }
                )
                mlflow.log_metric("train_samples", float(len(X_train)))
                mlflow.log_metric("positive_ratio", float(y_train.mean()))
                mlflow.log_metric("global_positive_ratio", float(target.mean()))
                mlflow.log_param("validation_size", validation_size)
                mlflow.log_param("threshold_recall_target", threshold_recall_target)
                mlflow.log_param("negative_downsample_ratio", downsample_ratio)
                if X_val is not None:
                    mlflow.log_metric("validation_samples", float(len(X_val)))

            pipeline.fit(X_train, y_train)

            y_proba = pipeline.predict_proba(X_test)[:, 1]

            decision_threshold = 0.5
            decision_threshold_precision = None
            decision_threshold_recall = None
            if X_val is not None:
                val_proba = pipeline.predict_proba(X_val)[:, 1]
                precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
                if len(thresholds) > 0:
                    precisions = precisions[:-1]
                    recalls = recalls[:-1]
                    candidates = [
                        (precisions[i], recalls[i], thresholds[i])
                        for i in range(len(thresholds))
                        if recalls[i] >= threshold_recall_target
                    ]
                    if candidates:
                        decision_threshold_precision, decision_threshold_recall, decision_threshold = max(
                            candidates,
                            key=lambda item: item[0],
                        )
                    else:
                        denom = np.clip(precisions + recalls, a_min=1e-12, a_max=None)
                        f1_vals = 2 * precisions * recalls / denom
                        best_index = int(np.argmax(f1_vals))
                        decision_threshold = float(thresholds[best_index])
                        decision_threshold_precision = float(precisions[best_index])
                        decision_threshold_recall = float(recalls[best_index])
                decision_threshold = float(decision_threshold)

            y_pred = (y_proba >= decision_threshold).astype(int)
            avg_precision = float(average_precision_score(y_test, y_proba))
            roc_auc = float(roc_auc_score(y_test, y_proba))

            metrics = {
                "avg_precision": avg_precision,
                "roc_auc": roc_auc,
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "decision_threshold": float(decision_threshold),
            }
            if decision_threshold_precision is not None:
                metrics["threshold_precision_at_target"] = float(decision_threshold_precision)
            if decision_threshold_recall is not None:
                metrics["threshold_recall_at_target"] = float(decision_threshold_recall)

            if log_to_mlflow:
                mlflow.log_metrics(metrics)

                signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
                log_model_kwargs = {
                    "sk_model": pipeline,
                    "artifact_path": "model",
                    "signature": signature,
                }
                if self.registered_model_name:
                    log_model_kwargs["registered_model_name"] = self.registered_model_name
                mlflow.sklearn.log_model(**log_model_kwargs)

        if persist_model:
            model_path = self._resolve_path(
                str(self.model_config.get("path", "/app/models/fraud_detection_model_logistic.pkl"))
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_path)
            logger.info("Training complete; model saved to %s", model_path)
        else:
            logger.info("Training complete; model persistence skipped per configuration.")

        return pipeline, metrics
