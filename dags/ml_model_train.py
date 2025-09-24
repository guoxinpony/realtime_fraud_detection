


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
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve, average_precision_score, precision_score, \
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
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_USERNAME', 'KAFKA_PASSWORD']
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


    


    



























