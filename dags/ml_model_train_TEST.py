


import json
import logging
import os

import boto3
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from kafka import KafkaConsumer
from mlflow.models import infer_signature
from numpy.array_api import astype
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve, \
                            average_precision_score, precision_score, \
                            recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Configure dual logging to file and stdout with structured format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    """
    End-to-end fraud detection training system implementing MLOps best practices.

    Key Architecture Components:
    - Configuration Management: Centralized YAML config with environment overrides
    - Data Ingestion: Kafka consumer with SASL/SSL authentication
    - Feature Engineering: Temporal, behavioral, and monetary feature constructs
    - Model Development: XGBoost with SMOTE for class imbalance
    - Hyperparameter Tuning: Randomized search with stratified cross-validation
    - Model Tracking: MLflow integration with metrics/artifact logging
    - Deployment Prep: Model serialization and registry

    The system is designed for horizontal scalability and cloud-native operation.
    """

    def __init__(self, config_path='/app/config.yaml'):
        # Environment hardening for containerized deployments
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        # Load environment variables before config to allow overrides
        load_dotenv(dotenv_path='/app/.env')

        # Configuration lifecycle management
        self.config = self._load_config(config_path)
        models_cfg = self.config.get('models', {})
        if 'xgboost' not in models_cfg:
            raise ValueError('Missing `models.xgboost` configuration section.')
        self.model_config = models_cfg['xgboost']

        # Security-conscious credential handling
        env_vars = {
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        }

        os.environ.update({
            k:v for k, v in env_vars.items() if v is not None
        })


        # Pre-flight system checks
        self._validate_environment()

        # MLflow configuration for experiment tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])


    def _load_config(self, config_path: str) -> dict:
        """
        Load and validate hierarchical configuration with fail-fast semantics.

        Implements:
        - YAML configuration parsing
        - Early validation of critical parameters
        - Audit logging of configuration loading
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully')
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    
    def _validate_environment(self):
        """
        System integrity verification with defense-in-depth checks:
        1. Required environment variables
        2. Object storage connectivity
        3. Credential validation

        Fails early to prevent partial initialization states.
        """
        security_protocol = self.config['kafka'].get('security_protocol', 'PLAINTEXT')
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS']
        if security_protocol.upper().startswith('SASL'):
            required_vars.extend(['KAFKA_USERNAME', 'KAFKA_PASSWORD'])

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f'Missing required environment variables: {missing}')


        self._check_minio_connection()


    def _check_minio_connection(self):
        """
        Validate object storage connectivity and bucket configuration.

        Implements:
        - S3 client initialization with error handling
        - Bucket existence check
        - Automatic bucket creation (if configured)

        Maintains separation of concerns between configuration and infrastructure setup.
        """
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=self.config['mlflow']['s3_endpoint_url'],
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('Minio connection verified. Buckets: %s', bucket_names)

            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')

            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info('Created missing MLFlow bucket: %s', mlflow_bucket)
        except Exception as e:
            logger.error('Minio connection failed: %s', str(e))


    def read_from_kafka(self) -> pd.DataFrame:
        """
        Secure Kafka consumer implementation with enterprise features:

        - SASL/SSL authentication
        - Auto-offset reset for recovery scenarios
        - Data quality checks:
          - Schema validation
          - Fraud label existence
          - Fraud rate monitoring

        Implements graceful shutdown on timeout/error conditions.
        """
        try:

            # read topic name from config
            topic = self.config['kafka']['topic']
            logger.info('Connecting to kafka topic %s', topic)

            # read configs 
            # build connection to kafka server and read all data into python env

            security_protocol = self.config['kafka'].get('security_protocol', 'PLAINTEXT').upper()
            sasl_kwargs = {}
            if security_protocol.startswith('SASL'):
                sasl_kwargs = {
                    'sasl_mechanism': self.config['kafka'].get('sasl_mechanism', 'PLAIN'),
                    'sasl_plain_username': self.config['kafka'].get('username'),
                    'sasl_plain_password': self.config['kafka'].get('password'),
                }
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config['kafka']['bootstrap_servers'].split(','),
                security_protocol=security_protocol,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                consumer_timeout_ms=self.config['kafka'].get('timeout', 10000),
                **sasl_kwargs,
            )

            # load and format messages into DataFrame, limmit to 30000 msgs

            messages = []
            max_messages = 10000
            for msg in consumer:
                messages.append(msg.value)
                if len(messages) >= max_messages:
                    break

            consumer.close()
            df = pd.DataFrame(messages)


            # data quality check: is empty dataframe
            if df.empty:
                raise ValueError('No messages received from Kafka.')

            # Temporal data standardization
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            max_rows = 20000
            if len(df) > max_rows:
                df = df.sort_values('timestamp').tail(max_rows).copy()

            # data quality check: fraud label is included in the data?
            if 'is_fraud' not in df.columns:
                raise ValueError('Fraud label (is_fraud) missing from Kafka data')

            # calculate percentage of fraud
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info('Kafka data read successfully with fraud rate: %.2f%%', fraud_rate)

            return df
        except Exception as e:
            logger.error('Failed to read data from Kafka: %s', str(e), exc_info=True)
            raise

  

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.sort_values(['user_id', 'timestamp']).copy()

        # ---- Temporal Feature Engineering ----
        # Captures time-based fraud patterns (e.g., nighttime transactions)
        df['transaction_hour'] = df['timestamp'].dt.hour
        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] < 5)).astype(int)
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        df['transaction_day'] = df['timestamp'].dt.day

        # -- Behavioral Feature Engineering --
        # Rolling window captures recent user activity patterns
        df['user_activity_24h'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: g.rolling('24h', on='timestamp', closed='left')['amount'].count().fillna(0)
        )

        # -- Monetary Feature Engineering --
        # Relative amount detection compared to user's historical pattern
        df['amount_to_avg_ratio'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: (g['amount'] / g['amount'].rolling(7, min_periods=1).mean()).fillna(1.0)
        )

        # -- Merchant Risk Profiling --
        # External risk intelligence integration point
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df['merchant_risk'] = df['merchant'].isin(high_risk_merchants).astype(int)

        feature_cols = [
            'amount', 'is_night', 'is_weekend', 'transaction_day', 'user_activity_24h',
            'amount_to_avg_ratio', 'merchant_risk', 'merchant'
        ]

        # Schema validation guard
        if 'is_fraud' not in df.columns:
            raise ValueError('Missing target column "is_fraud"')

        return df[feature_cols + ['is_fraud']]


    def train_model(self):
        
        try:
            logger.info('Starting model training')

            # read data from kafka server
            df = self.read_from_kafka()

            # feature engineering of data
            data = self.create_features(df)
            
            # Train/Test split with stratification
            X = data.drop(columns=['is_fraud'])
            y = data['is_fraud']

            # Class imbalance safeguards
            if y.sum() == 0:
                raise ValueError('No positive samples in training data')
            if y.sum() < 10:
                logger.warning('Low positive samples: %d. Consider additional data augmentation', y.sum())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.model_config.get('test_size', 0.2),
                stratify=y,
                random_state=self.model_config.get('seed', 42)
            )


            # MLflow experiment tracking context
            with mlflow.start_run():
                # Dataset metadata logging:
                # number of training samples, number of fraud samples
                # percentage of fraud samples, number of test samples
                mlflow.log_metrics({
                    'train_samples': X_train.shape[0],
                    'positive_samples': int(y_train.sum()),
                    'class_ratio': float(y_train.mean()),
                    'test_samples': X_test.shape[0]
                })

                # Categorical feature preprocessing
                preprocessor = ColumnTransformer([
                    ('merchant_encoder', OrdinalEncoder(
                        handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32
                    ), ['merchant'])
                ], remainder='passthrough')

                # XGBoost configuration with efficiency optimizations

                fraud_ratio = max(y_train.mean(), 1e-6)
                scale_pos_weight = (1 - fraud_ratio) / fraud_ratio

                xgb = XGBClassifier(
                    eval_metric='aucpr',  # Optimizes for precision-recall area
                    random_state=self.model_config.get('seed', 42),
                    reg_lambda=1.0,
                    n_estimators=self.model_config['params']['n_estimators'],
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    tree_method=self.model_config.get('tree_method', 'hist')  # GPU-compatible
                )
                
                # Imbalanced learning pipeline
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    # ('smote', SMOTE(sampling_strategy=0.3,random_state=self.model_config.get('seed', 42))),
                    ('classifier', xgb)
                ])


                # Hyperparameter search space design, grid search, try each combination of 
                #  hyperparameters
                param_dist = {
                    'classifier__max_depth': [3, 5],  # Depth control for regularization
                    'classifier__learning_rate': [0.01, 0.1],  # Conservative range
                    'classifier__subsample': [0.6, 1.0],  # Stochastic gradient boosting
                    'classifier__colsample_bytree': [0.6, 1.0],  # Feature randomization
                    'classifier__gamma': [0, 0.1],  # Complexity cont
                    'classifier__reg_alpha': [0, 0.5]  # L1 regularization
                }


                # Optimizing for F-beta score (beta=2 emphasizes recall)
                searcher = RandomizedSearchCV(
                    pipeline,
                    param_dist,
                    n_iter=5,
                    scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
                    cv=StratifiedKFold(n_splits=3, shuffle=True),
                    n_jobs=-1,
                    refit=True,
                    error_score='raise',
                    random_state=self.model_config.get('seed', 42)
                )

                # find best model and hyperparameters
                logger.info('Starting hyperparameter tuning...')
                searcher.fit(X_train, y_train)
                best_model = searcher.best_estimator_
                best_params = searcher.best_params_
                logger.info('Best hyperparameters: %s', best_params)


                # Threshold optimization using training data
                train_proba = best_model.predict_proba(X_train)[:, 1]
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_train, train_proba)
                f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in
                             zip(precision_arr[:-1], recall_arr[:-1])]
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info('Optimal threshold determined: %.4f', best_threshold)


                # Model evaluation
                X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)
                test_proba = best_model.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]
                y_pred = (test_proba >= best_threshold).astype(int)


                # Comprehensive metrics suite
                metrics = {
                    'auc_pr': float(average_precision_score(y_test, test_proba)),
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                    'threshold': float(best_threshold)
                }
                
                # record metrics and best hyperparameter combinations in mlflow
                mlflow.log_metrics(metrics)
                mlflow.log_params(best_params)


                # Confusion matrix visualization and record it into mlflow
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Not Fraud', 'Fraud'])
                plt.yticks(tick_marks, ['Not Fraud', 'Fraud'])

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')

                plt.tight_layout()
                cm_filename = 'confusion_matrix.png'
                plt.savefig(cm_filename)
                mlflow.log_artifact(cm_filename)
                plt.close()


                # Precision-Recall curve documentation in mlflow
                plt.figure(figsize=(10, 6))
                plt.plot(recall_arr, precision_arr, marker='.', label='Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                pr_filename = 'precision_recall_curve.png'
                plt.savefig(pr_filename)
                mlflow.log_artifact(pr_filename)
                plt.close()


                # Model packaging and registry
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path='model',
                    signature=signature,
                    registered_model_name='fraud_detection_model_TEST'
                )

                # Model serialization for deployment
                model_path = self.model_config.get(
                    'test_path',
                    self.model_config.get('path', '/app/models/fraud_detection_model_TEST.pkl')
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(best_model, model_path)

                logger.info('Training successfully completed with metrics: %s', metrics)

                return best_model, metrics



        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise e


    


    



























