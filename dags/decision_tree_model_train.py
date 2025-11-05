########################################################
# developed by Yikai Cai, 2025-11-05
########################################################

import json
import logging
import os
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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DecisionTreeFraudTrainer:
    """Train a Decision Tree classifier for fraud detection using Kafka sourced data."""

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
        self.model_config: Dict[str, object] = models_config.get("decision_tree", {})
        if not self.model_config:
            raise ValueError("Configuration is missing `models.decision_tree` settings.")

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

        required = [
            "transaction_time",
            "amount",
            "merchant",
            "category",
            "city",
            "state",
            "zip",
            "lat",
            "long",
            "city_pop",
            "merchant_lat",
            "merchant_long",
            "is_fraud",
        ]
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
        return df

    def train_model(self) -> Tuple[Pipeline, Dict[str, float]]:
        logger.info("Starting Decision Tree fraud detection training workflow.")
        df = self.read_from_kafka()
        df = self._augment_features(df)

        features = df.drop(columns=["is_fraud", "transaction_time"])
        target = df["is_fraud"]

        if target.sum() == 0:
            raise ValueError("No fraud samples present in the training data.")

        seed = int(self.model_config.get("seed", 42))

        features = features.reset_index(drop=True)
        target = target.reset_index(drop=True)

        preprocessor = self._build_feature_pipeline()

        def _build_pipeline() -> Pipeline:
            model = DecisionTreeClassifier(**params)
            return Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

        params = dict(self.model_config.get("params", {}))
        params.setdefault("class_weight", "balanced")
        params.setdefault("random_state", seed)

        with mlflow.start_run():
            mlflow.log_params({f"classifier__{k}": v for k, v in params.items()})
            mlflow.log_metric("train_samples", float(len(features)))
            mlflow.log_metric("positive_ratio", float(target.mean()))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            fold_metrics = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(features, target), start=1):
                X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

                pipeline = _build_pipeline()
                pipeline.fit(X_train, y_train)

                y_proba = pipeline.predict_proba(X_val)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)

                metrics = {
                    "avg_precision": float(average_precision_score(y_val, y_proba)),
                    "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_val, y_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_val, y_proba)),
                }
                fold_metrics.append(metrics)

                for name, value in metrics.items():
                    mlflow.log_metric(f"{name}_fold_{fold}", value)

            mean_metrics = {name: float(np.mean([m[name] for m in fold_metrics])) for name in fold_metrics[0]}
            mlflow.log_metrics({f"{name}_mean": value for name, value in mean_metrics.items()})

            final_pipeline = _build_pipeline()
            final_pipeline.fit(features, target)

            signature = infer_signature(features, final_pipeline.predict_proba(features)[:, 1])
            log_model_kwargs = {
                "sk_model": final_pipeline,
                "artifact_path": "model",
                "signature": signature,
            }
            if self.registered_model_name:
                log_model_kwargs["registered_model_name"] = self.registered_model_name
            mlflow.sklearn.log_model(**log_model_kwargs)

        default_path = "/app/models/fraud_detection_model_decision_tree.pkl"
        model_path = self._resolve_path(str(self.model_config.get("path", default_path)))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, model_path)
        logger.info("Training complete; model saved to %s", model_path)

        return final_pipeline, mean_metrics
