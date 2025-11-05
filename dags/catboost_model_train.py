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
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from kafka import KafkaConsumer
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CatBoostFraudTrainer:
    """Train a CatBoost classifier for fraud detection using Kafka sourced data."""

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
        self.model_config: Dict[str, object] = models_config.get("catboost", {})
        if not self.model_config:
            raise ValueError("Configuration is missing `models.catboost` settings.")

        self.registered_model_name = self.model_config.get(
            "mlflow_registered_model_name",
            self.config.get("mlflow", {}).get("registered_model_name"),
        )

        # Allow CLI override while keeping config defaults.
        self.max_messages = sample_size or int(self.kafka_config.get("max_messages", 20000))

        os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", str(self.kafka_config.get("bootstrap_servers", "")))

        # Ensure MLflow is initialised according to config.
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
    def _build_feature_pipeline(numeric_features: list) -> ColumnTransformer:
        """Build preprocessing pipeline for numeric features only.
        CatBoost handles categorical features internally, so we only need to preprocess numeric features.
        Note: We do NOT use remainder="passthrough" to avoid column count mismatch.
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
            ],
            remainder="drop",  # Drop non-numeric features - we'll add them back manually
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
        df["merchant_distance"] = np.sqrt(
            (df["lat"] - df["merchant_lat"]) ** 2 + (df["long"] - df["merchant_long"]) ** 2
        )
        df["city_pop_log"] = np.log1p(df["city_pop"].clip(lower=0))
        return df

    @staticmethod
    def _get_categorical_features(feature_cols: list) -> list:
        """Get list of categorical feature column indices for CatBoost."""
        categorical_features = ["category", "state", "gender", "city", "merchant"]
        
        # Find indices of categorical features
        cat_indices = [i for i, col in enumerate(feature_cols) if col in categorical_features]
        return cat_indices

    def train_model(
        self,
        df: Optional[pd.DataFrame] = None,
        params_override: Optional[Dict[str, float]] = None,
        run_name: Optional[str] = None,
        log_to_mlflow: bool = True,
        persist_model: bool = True,
    ) -> Tuple[Pipeline, Dict[str, float]]:
        logger.info("Starting CatBoost fraud detection training workflow with 10-fold cross-validation.")
        if df is None:
            df = self.read_from_kafka()
        else:
            df = df.copy()
        df = self._augment_features(df)
        
        # Add noise to numeric features to simulate real-world uncertainty
        # This helps prevent overfitting and reduces inflated AUC
        numeric_cols = ["amount", "zip", "lat", "long", "city_pop", "merchant_lat", "merchant_long",
                       "transaction_hour", "transaction_dayofweek", "transaction_month",
                       "amount_log", "merchant_distance", "city_pop_log"]
        noise_scale = 0.05  # 5% noise - further increased to reduce overfitting and inflated AUC
        for col in numeric_cols:
            if col in df.columns:
                std_val = df[col].std()
                if std_val > 0:
                    noise = np.random.normal(0, std_val * noise_scale, size=len(df))
                    df[col] = df[col] + noise
        logger.info("Added %.1f%% noise to numeric features to reduce overfitting", noise_scale * 100)

        features = df.drop(columns=["is_fraud", "transaction_time"])
        target = df["is_fraud"]

        if target.sum() == 0:
            raise ValueError("No fraud samples present in the training data.")

        test_size = float(self.model_config.get("test_size", 0.2))
        n_folds = int(self.model_config.get("n_folds", 10))
        early_stopping_rounds = int(self.model_config.get("early_stopping_rounds", 25))
        threshold_recall_target = float(self.model_config.get("threshold_recall_target", 0.95))
        seed = int(self.model_config.get("seed", 42))
        
        # Feature selection parameters
        use_feature_selection = bool(self.model_config.get("use_feature_selection", True))
        feature_selection_threshold = float(self.model_config.get("feature_selection_threshold", 0.01))  # Keep features with importance > threshold
        max_features = int(self.model_config.get("max_features", 10))  # Maximum number of features to keep

        # Split data into train (for CV) and test (for final evaluation)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            stratify=target,
            random_state=seed,
        )

        logger.info("Data split: %d samples for cross-validation, %d samples for final test", 
                   len(X_train_full), len(X_test))

        params = dict(self.model_config.get("params", {}))
        if params_override:
            params.update(params_override)
        params.setdefault("random_seed", seed)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        # Add very strong regularization and constraints to prevent overfitting and reduce inflated AUC
        params.setdefault("l2_leaf_reg", max(params.get("l2_leaf_reg", 20), 20))  # Very strong L2 regularization
        params.setdefault("depth", min(params.get("depth", 4), 3))  # Very shallow trees
        params.setdefault("max_ctr_complexity", 1)  # Limit categorical feature combinations
        params.setdefault("one_hot_max_size", 0)  # Disable one-hot encoding
        params.setdefault("bootstrap_type", "Bernoulli")  # Use Bernoulli bootstrap
        params.setdefault("subsample", 0.6)  # Very aggressive row sampling
        params.setdefault("random_strength", max(params.get("random_strength", 2.0), 2.0))  # More randomness to scoring
        params.setdefault("min_data_in_leaf", max(params.get("min_data_in_leaf", 10), 10))  # Much higher minimum samples
        params.setdefault("rsm", 0.6)  # Random subspace method - feature sampling
        # Note: bagging_temperature only works with Bayesian bootstrap, removed for Bernoulli

        run_ctx = (
            mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
            if log_to_mlflow
            else nullcontext()
        )

        with run_ctx:
            if log_to_mlflow:
                mlflow.log_params({f"classifier__{k}": v for k, v in params.items()})
                mlflow.log_metric("train_samples", float(len(X_train_full)))
                mlflow.log_metric("test_samples", float(len(X_test)))
                mlflow.log_metric("positive_ratio", float(y_train_full.mean()))
                mlflow.log_metric("global_positive_ratio", float(target.mean()))
                mlflow.log_param("n_folds", n_folds)
                mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
                mlflow.log_param("threshold_recall_target", threshold_recall_target)
                mlflow.log_param("use_feature_selection", use_feature_selection)
                if use_feature_selection:
                    mlflow.log_param("feature_selection_threshold", feature_selection_threshold)
                    mlflow.log_param("max_features", max_features)

            # Feature selection: train a preliminary model to get feature importance
            selected_features = None
            if use_feature_selection:
                logger.info("Performing feature selection...")
                # Get actual available features from DataFrame
                all_possible_numeric = ["amount", "zip", "lat", "long", "city_pop", "merchant_lat", "merchant_long",
                                       "transaction_hour", "transaction_dayofweek", "transaction_month",
                                       "amount_log", "merchant_distance", "city_pop_log"]
                all_possible_categorical = ["category", "state", "gender", "city", "merchant"]
                
                # Only use features that actually exist in the DataFrame
                numeric_features_list = [f for f in all_possible_numeric if f in X_train_full.columns]
                categorical_features_list = [f for f in all_possible_categorical if f in X_train_full.columns]
                
                logger.info("Available numeric features: %s", numeric_features_list)
                logger.info("Available categorical features: %s", categorical_features_list)
                
                # Train a quick model on full training data to get feature importance
                if numeric_features_list:
                    preprocessor_selection = self._build_feature_pipeline(numeric_features_list)
                    preprocessor_selection.fit(X_train_full)
                    X_train_processed = preprocessor_selection.transform(X_train_full)
                    
                    # Reconstruct DataFrame with processed numeric features and original categorical features
                    X_train_numeric = pd.DataFrame(
                        X_train_processed,
                        columns=numeric_features_list,
                        index=X_train_full.index
                    )
                    if categorical_features_list:
                        X_train_processed = pd.concat([
                            X_train_numeric,
                            X_train_full[categorical_features_list]
                        ], axis=1)
                    else:
                        X_train_processed = X_train_numeric
                else:
                    # No numeric features, use DataFrame as-is
                    X_train_processed = X_train_full.copy()
                
                all_features_list = numeric_features_list + categorical_features_list
                if len(all_features_list) == 0:
                    raise ValueError("No features available for training. Please check your data.")
                cat_indices_selection = self._get_categorical_features(all_features_list) if categorical_features_list else []
                
                # Create a simple model for feature selection
                selection_params = params.copy()
                selection_params.pop("scale_pos_weight", None)
                fraud_ratio_selection = y_train_full.mean()
                if fraud_ratio_selection > 0:
                    scale_pos_weight = (1 - fraud_ratio_selection) / fraud_ratio_selection
                    selection_params["class_weights"] = [1.0, scale_pos_weight]
                else:
                    selection_params["class_weights"] = [1.0, 1.0]
                
                # Override specific parameters for feature selection model
                selection_params.update({
                    "iterations": 20,
                    "depth": 3,
                    "learning_rate": 0.1,
                    "cat_features": cat_indices_selection,
                    "random_seed": seed,
                    "verbose": False,
                    "allow_writing_files": False,
                })
                selection_model = CatBoostClassifier(**selection_params)
                selection_model.fit(X_train_processed, y_train_full, verbose=False)
                
                # Get feature importance
                feature_importance = selection_model.feature_importances_
                feature_names = all_features_list
                
                # Create DataFrame with feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                logger.info("Top 10 features by importance:\n%s", importance_df.head(10).to_string())
                
                # Select features based on threshold and max_features
                if feature_selection_threshold > 0:
                    selected_features = importance_df[importance_df['importance'] >= feature_selection_threshold]['feature'].tolist()
                else:
                    selected_features = importance_df.head(max_features)['feature'].tolist()
                
                # Ensure we don't exceed max_features
                if len(selected_features) > max_features:
                    selected_features = importance_df.head(max_features)['feature'].tolist()
                
                logger.info("Selected %d features out of %d: %s", 
                           len(selected_features), len(feature_names), selected_features)
                
                if log_to_mlflow:
                    mlflow.log_metric("selected_features_count", float(len(selected_features)))
                    mlflow.log_param("selected_features", ",".join(selected_features))
                
                # Update feature lists for preprocessing
                X_train_full = X_train_full[selected_features]
                X_test = X_test[selected_features]
            
            # 10-fold cross-validation
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            cv_metrics = {
                "avg_precision": [],
                "roc_auc": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "decision_threshold": [],
                "threshold_precision_at_target": [],
                "threshold_recall_at_target": [],
                "best_iteration": [],
            }

            logger.info("Starting %d-fold cross-validation...", n_folds)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
                logger.info("Processing fold %d/%d", fold_idx, n_folds)
                X_train_fold = X_train_full.iloc[train_idx]
                X_val_fold = X_train_full.iloc[val_idx]
                y_train_fold = y_train_full.iloc[train_idx]
                y_val_fold = y_train_full.iloc[val_idx]

                # Define feature lists (use selected features if feature selection was performed)
                if use_feature_selection and selected_features:
                    all_available_features = ["amount", "zip", "lat", "long", "city_pop", "merchant_lat", "merchant_long",
                                            "transaction_hour", "transaction_dayofweek", "transaction_month",
                                            "amount_log", "merchant_distance", "city_pop_log",
                                            "category", "state", "gender", "city", "merchant"]
                    numeric_features = [f for f in selected_features if f in ["amount", "zip", "lat", "long", "city_pop", 
                                                                              "merchant_lat", "merchant_long",
                                                                              "transaction_hour", "transaction_dayofweek", 
                                                                              "transaction_month", "amount_log", 
                                                                              "merchant_distance", "city_pop_log"]]
                    categorical_features = [f for f in selected_features if f in ["category", "state", "gender", "city", "merchant"]]
                else:
                    numeric_features = ["amount", "zip", "lat", "long", "city_pop", "merchant_lat", "merchant_long",
                                        "transaction_hour", "transaction_dayofweek", "transaction_month",
                                        "amount_log", "merchant_distance", "city_pop_log"]
                    categorical_features = ["category", "state", "gender", "city", "merchant"]
                
                # Fit preprocessor on fold training data (only numeric features)
                preprocessor_fold = self._build_feature_pipeline(numeric_features)
                preprocessor_fold.fit(X_train_fold)
                X_train_fold_processed = preprocessor_fold.transform(X_train_fold)
                X_val_fold_processed = preprocessor_fold.transform(X_val_fold)

                # Reconstruct DataFrame with processed numeric features and original categorical features
                X_train_fold_numeric = pd.DataFrame(
                    X_train_fold_processed,
                    columns=numeric_features,
                    index=X_train_fold.index
                )
                # Only include categorical features that exist in the DataFrame
                existing_categorical = [f for f in categorical_features if f in X_train_fold.columns]
                if existing_categorical:
                    X_train_fold_processed = pd.concat([
                        X_train_fold_numeric,
                        X_train_fold[existing_categorical]
                    ], axis=1)
                else:
                    X_train_fold_processed = X_train_fold_numeric
                
                X_val_fold_numeric = pd.DataFrame(
                    X_val_fold_processed,
                    columns=numeric_features,
                    index=X_val_fold.index
                )
                # Only include categorical features that exist in the DataFrame
                if existing_categorical:
                    X_val_fold_processed = pd.concat([
                        X_val_fold_numeric,
                        X_val_fold[existing_categorical]
                    ], axis=1)
                else:
                    X_val_fold_processed = X_val_fold_numeric
                
                # Get categorical feature indices for processed data
                # Use existing categorical features (may be empty if none were selected)
                all_features = numeric_features + (existing_categorical if existing_categorical else [])
                cat_indices_fold = self._get_categorical_features(all_features)

                # Calculate class weights for this fold
                fraud_ratio_fold = y_train_fold.mean()
                fold_params = params.copy()
                # Remove scale_pos_weight if present (CatBoost doesn't allow both)
                fold_params.pop("scale_pos_weight", None)
                if fraud_ratio_fold > 0:
                    # CatBoost uses class_weights parameter
                    scale_pos_weight = (1 - fraud_ratio_fold) / fraud_ratio_fold
                    fold_params["class_weights"] = [1.0, scale_pos_weight]
                else:
                    fold_params["class_weights"] = [1.0, 1.0]

                # Create model for this fold
                model_fold = CatBoostClassifier(
                    cat_features=cat_indices_fold,
                    **fold_params
                )

                # Set up early stopping
                fit_kwargs = {}
                if early_stopping_rounds > 0:
                    fit_kwargs["eval_set"] = (X_val_fold_processed, y_val_fold)
                    fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                    fit_kwargs["verbose"] = False

                # Train model on fold training data
                model_fold.fit(X_train_fold_processed, y_train_fold, **fit_kwargs)
                
                # Record best iteration if available
                if hasattr(model_fold, "best_iteration_") and model_fold.best_iteration_ is not None:
                    cv_metrics["best_iteration"].append(float(model_fold.best_iteration_))

                # Predict on fold validation data
                val_proba = model_fold.predict_proba(X_val_fold_processed)[:, 1]

                # Find optimal threshold on validation set
                decision_threshold = 0.5
                decision_threshold_precision = None
                decision_threshold_recall = None
                precisions, recalls, thresholds = precision_recall_curve(y_val_fold, val_proba)
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

                # Evaluate on fold validation data
                y_pred_fold = (val_proba >= decision_threshold).astype(int)
                cv_metrics["avg_precision"].append(float(average_precision_score(y_val_fold, val_proba)))
                cv_metrics["roc_auc"].append(float(roc_auc_score(y_val_fold, val_proba)))
                cv_metrics["precision"].append(float(precision_score(y_val_fold, y_pred_fold, zero_division=0)))
                cv_metrics["recall"].append(float(recall_score(y_val_fold, y_pred_fold, zero_division=0)))
                cv_metrics["f1"].append(float(f1_score(y_val_fold, y_pred_fold, zero_division=0)))
                cv_metrics["decision_threshold"].append(float(decision_threshold))
                if decision_threshold_precision is not None:
                    cv_metrics["threshold_precision_at_target"].append(float(decision_threshold_precision))
                if decision_threshold_recall is not None:
                    cv_metrics["threshold_recall_at_target"].append(float(decision_threshold_recall))

                logger.info("Fold %d: F1=%.4f, Precision=%.4f, Recall=%.4f, ROC-AUC=%.4f",
                           fold_idx, cv_metrics["f1"][-1], cv_metrics["precision"][-1],
                           cv_metrics["recall"][-1], cv_metrics["roc_auc"][-1])

            # Calculate mean metrics across folds
            metrics = {
                "cv_avg_precision_mean": float(np.mean(cv_metrics["avg_precision"])),
                "cv_avg_precision_std": float(np.std(cv_metrics["avg_precision"])),
                "cv_roc_auc_mean": float(np.mean(cv_metrics["roc_auc"])),
                "cv_roc_auc_std": float(np.std(cv_metrics["roc_auc"])),
                "cv_precision_mean": float(np.mean(cv_metrics["precision"])),
                "cv_precision_std": float(np.std(cv_metrics["precision"])),
                "cv_recall_mean": float(np.mean(cv_metrics["recall"])),
                "cv_recall_std": float(np.std(cv_metrics["recall"])),
                "cv_f1_mean": float(np.mean(cv_metrics["f1"])),
                "cv_f1_std": float(np.std(cv_metrics["f1"])),
                "cv_decision_threshold_mean": float(np.mean(cv_metrics["decision_threshold"])),
            }
            
            if cv_metrics["threshold_precision_at_target"]:
                metrics["cv_threshold_precision_at_target_mean"] = float(np.mean(cv_metrics["threshold_precision_at_target"]))
            if cv_metrics["threshold_recall_at_target"]:
                metrics["cv_threshold_recall_at_target_mean"] = float(np.mean(cv_metrics["threshold_recall_at_target"]))
            if cv_metrics["best_iteration"]:
                metrics["cv_best_iteration_mean"] = float(np.mean(cv_metrics["best_iteration"]))

            logger.info("Cross-validation completed. Mean F1=%.4f±%.4f, Mean Precision=%.4f±%.4f, Mean Recall=%.4f±%.4f",
                       metrics["cv_f1_mean"], metrics["cv_f1_std"],
                       metrics["cv_precision_mean"], metrics["cv_precision_std"],
                       metrics["cv_recall_mean"], metrics["cv_recall_std"])

            # Train final model on all training data for persistence
            logger.info("Training final model on all training data...")
            # Use same feature lists as in cross-validation
            all_possible_numeric_final = ["amount", "zip", "lat", "long", "city_pop", "merchant_lat", "merchant_long",
                                         "transaction_hour", "transaction_dayofweek", "transaction_month",
                                         "amount_log", "merchant_distance", "city_pop_log"]
            all_possible_categorical_final = ["category", "state", "gender", "city", "merchant"]
            
            if use_feature_selection and selected_features:
                # Use selected features, but filter to only those that exist in DataFrame
                numeric_features = [f for f in selected_features if f in all_possible_numeric_final and f in X_train_full.columns]
                categorical_features = [f for f in selected_features if f in all_possible_categorical_final and f in X_train_full.columns]
            else:
                # Use all available features from DataFrame
                numeric_features = [f for f in all_possible_numeric_final if f in X_train_full.columns]
                categorical_features = [f for f in all_possible_categorical_final if f in X_train_full.columns]
            
            logger.info("Final model numeric features: %s", numeric_features)
            logger.info("Final model categorical features: %s", categorical_features)
            
            if not numeric_features:
                raise ValueError("No numeric features available for final model training.")
            
            preprocessor = self._build_feature_pipeline(numeric_features)
            preprocessor.fit(X_train_full)
            X_train_full_processed = preprocessor.transform(X_train_full)
            
            # Reconstruct DataFrame with processed numeric features and original categorical features
            
            X_train_full_numeric = pd.DataFrame(
                X_train_full_processed,
                columns=numeric_features,
                index=X_train_full.index
            )
            # Only include categorical features that exist in the DataFrame
            existing_categorical_final = [f for f in categorical_features if f in X_train_full.columns]
            if existing_categorical_final:
                X_train_full_processed = pd.concat([
                    X_train_full_numeric,
                    X_train_full[existing_categorical_final]
                ], axis=1)
            else:
                X_train_full_processed = X_train_full_numeric
            
            X_test_processed_numeric = preprocessor.transform(X_test)
            X_test_processed_numeric = pd.DataFrame(
                X_test_processed_numeric,
                columns=numeric_features,
                index=X_test.index
            )
            # Only include categorical features that exist in the DataFrame
            if existing_categorical_final:
                X_test_processed = pd.concat([
                    X_test_processed_numeric,
                    X_test[existing_categorical_final]
                ], axis=1)
            else:
                X_test_processed = X_test_processed_numeric
            
            all_features = numeric_features + categorical_features
            
            fraud_ratio = y_train_full.mean()
            final_params = params.copy()
            # Remove scale_pos_weight if present (CatBoost doesn't allow both)
            final_params.pop("scale_pos_weight", None)
            if fraud_ratio > 0:
                scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
                final_params["class_weights"] = [1.0, scale_pos_weight]
            else:
                final_params["class_weights"] = [1.0, 1.0]

            # Get categorical feature indices for processed data
            # Use existing categorical features (may be empty if none were selected)
            all_features_final = numeric_features + (existing_categorical_final if existing_categorical_final else [])
            cat_indices_final = [i for i, col in enumerate(all_features_final) if col in existing_categorical_final]

            final_model = CatBoostClassifier(
                cat_features=cat_indices_final,
                **final_params
            )
            
            # Use early stopping with a validation set created from training data
            if early_stopping_rounds > 0:
                # Create a small validation set from training data for early stopping
                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                    X_train_full_processed,
                    y_train_full,
                    test_size=0.1,
                    stratify=y_train_full,
                    random_state=seed,
                )
                final_model.fit(
                    X_train_final, y_train_final,
                    eval_set=(X_val_final, y_val_final),
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                if log_to_mlflow and hasattr(final_model, "best_iteration_") and final_model.best_iteration_ is not None:
                    mlflow.log_metric("best_iteration", float(final_model.best_iteration_))
            else:
                final_model.fit(X_train_full_processed, y_train_full, verbose=False)

            # Create pipeline for MLflow and persistence
            # Note: CatBoost doesn't work well with sklearn Pipeline, so we'll use a custom wrapper
            # For simplicity, we'll save the model and preprocessor separately
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", final_model),
                ]
            )

            # Evaluate final model on test set
            test_proba = final_model.predict_proba(X_test_processed)[:, 1]
            final_decision_threshold = metrics["cv_decision_threshold_mean"]
            test_pred = (test_proba >= final_decision_threshold).astype(int)
            
            test_metrics = {
                "test_avg_precision": float(average_precision_score(y_test, test_proba)),
                "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
                "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
            }
            metrics.update(test_metrics)
            metrics["decision_threshold"] = final_decision_threshold

            logger.info("Final test set evaluation: F1=%.4f, Precision=%.4f, Recall=%.4f, ROC-AUC=%.4f",
                       test_metrics["test_f1"], test_metrics["test_precision"],
                       test_metrics["test_recall"], test_metrics["test_roc_auc"])

            if log_to_mlflow:
                mlflow.log_metrics(metrics)

                signature = infer_signature(X_train_full_processed, final_model.predict_proba(X_train_full_processed)[:, 1])
                log_model_kwargs = {
                    "sk_model": pipeline,
                    "artifact_path": "model",
                    "signature": signature,
                }
                if self.registered_model_name:
                    log_model_kwargs["registered_model_name"] = self.registered_model_name
                mlflow.sklearn.log_model(**log_model_kwargs)

        if persist_model:
            model_path = self._resolve_path(str(self.model_config.get("path", "/app/models/fraud_detection_model_catboost.pkl")))
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_path)
            logger.info("Training complete; model saved to %s", model_path)
        else:
            logger.info("Training complete; model persistence skipped per configuration.")

        return pipeline, metrics

