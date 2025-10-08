"""
Real-time Fraud Detection Inference Pipeline

This script consumes transaction data from Kafka, processes it using Spark Streaming,
applies a pre-trained machine learning model to detect fraudulent transactions,
and writes predictions back to Kafka.
"""

# Standard library imports
import logging
import os

# Third-party imports
import joblib  # For loading serialized ML models
import yaml  # For parsing YAML configuration files
from dotenv import load_dotenv  # For loading environment variables from .env file

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json,
    col,
    hour,
    dayofmonth,
    dayofweek,
    month,
    when,
    lit,
    coalesce,
    to_timestamp,
    from_unixtime,
    to_json,
    struct,
)
from pyspark.sql.pandas.functions import pandas_udf  # For Pandas vectorized UDFs
from pyspark.sql.types import (StructType, StructField, StringType,
                              IntegerType, DoubleType, TimestampType)

# Configure logging to track pipeline operations and errors
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO for operational messages
    format="%(asctime)s [%(levelname)s] %(message)s"  # Structured log format
)
logger = logging.getLogger(__name__)  # Create logger instance for the module


class FraudDetectionInference:
    """
    Fraud detection inference pipeline class that handles:
    - Configuration loading
    - Spark session management
    - Kafka stream processing
    - Feature engineering
    - Model inference
    - Results publishing

    Attributes:
        config (dict): Pipeline configuration parameters
        spark (SparkSession): Spark session instance
        model: Loaded ML model for fraud detection
        broadcast_model: Model broadcast to Spark workers for distributed inference
    """

    # Class variables for Kafka configuration
    bootstrap_servers = None
    topic = None
    security_protocol = None
    sasl_mechanism = None
    username = None
    password = None
    sasl_jaas_config = None

    def __init__(self, config_path="/app/config.yaml"):
        """Initialize pipeline with configuration and dependencies

        Args:
            config_path (str): Path to YAML configuration file
        """
        # Load environment variables from .env file
        load_dotenv(dotenv_path="/app/.env")

        # Load pipeline configuration from YAML file
        self.config = self._load_config(config_path)
        models_config = self.config.get("models", {})
        self.model_config = models_config.get("xgboost")
        if not self.model_config:
            raise ValueError("Missing `models.xgboost` configuration section.")

        self.prediction_threshold = float(self.model_config.get("prediction_threshold", 0.5))

        # Initialize Spark session with Kafka integration packages
        self.spark = self._init_spark_session()

        # Load and broadcast ML model to worker nodes for distributed inference
        self.model = self._load_model(self.model_config["path"])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)

        # Debug: Log loaded environment variables (sensitive values should be masked in production)
        logger.debug("Environment variables loaded: %s", dict(os.environ))

    def _load_model(self, model_path):
        """Load pre-trained fraud detection model from disk

        Args:
            model_path (str): Path to serialized model file

        Returns:
            model: Loaded ML model

        Raises:
            Exception: If model loading fails
        """
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
            return model
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise

    @staticmethod
    def _load_config(config_path):
        """Load YAML configuration file

        Args:
            config_path (str): Path to configuration file

        Returns:
            dict: Parsed configuration parameters

        Raises:
            Exception: If file loading or parsing fails
        """
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _init_spark_session(self):
        """Initialize Spark session with Kafka dependencies

        Returns:
            SparkSession: Configured Spark session

        Raises:
            Exception: If Spark initialization fails
        """
        try:
            # Get required packages from config (typically Kafka integration packages)
            packages = self.config.get("spark", {}).get("packages", "")

            # Build Spark session with application name and packages
            builder = SparkSession.builder.appName("FraudDetectionInferenceStreaming")

            # Add Maven packages if specified in config
            if packages:
                builder = builder.config("spark.jars.packages", packages)

            spark = builder.getOrCreate()
            logger.info("Spark Session initialized.")
            return spark
        except Exception as e:
            logger.error("Error initializing Spark Session: %s", str(e))
            raise

    def read_from_kafka(self):
        """Read streaming data from Kafka topic and parse JSON payload

        Returns:
            DataFrame: Spark DataFrame containing parsed transaction data
        """
        kafka_config = self.config["kafka"]
        input_topic = kafka_config.get("test_data", kafka_config.get("topic", "fraud_test_data"))
        logger.info("Reading data from Kafka topic %s", input_topic)

        # Load Kafka configuration parameters with fallback values
        kafka_bootstrap_servers = kafka_config.get("bootstrap_servers", "localhost:19092")
        kafka_topic = input_topic
        kafka_security_protocol = kafka_config.get("security_protocol", "PLAINTEXT").upper()
        kafka_sasl_mechanism = kafka_config.get("sasl_mechanism", "PLAIN")
        kafka_username = kafka_config.get("username")
        kafka_password = kafka_config.get("password")

        kafka_options = {
            "kafka.bootstrap.servers": kafka_bootstrap_servers,
            "subscribe": kafka_topic,
            "startingOffsets": "latest",
            "kafka.security.protocol": kafka_security_protocol,
        }

        # Configure Kafka SASL authentication string
        if kafka_security_protocol.startswith("SASL"):
            kafka_options.update({
                "kafka.sasl.mechanism": kafka_sasl_mechanism,
                "kafka.sasl.jaas.config": (
                    f'org.apache.kafka.common.security.plain.PlainLoginModule required '
                    f'username="{kafka_username}" password="{kafka_password}";'
                ),
            })

        # Store Kafka configuration in instance variables for reuse
        self.bootstrap_servers = kafka_bootstrap_servers
        self.topic = kafka_topic
        self.security_protocol = kafka_security_protocol
        self.sasl_mechanism = kafka_sasl_mechanism
        self.username = kafka_username
        self.password = kafka_password
        self.sasl_jaas_config = kafka_options.get("kafka.sasl.jaas.config")

        # Define schema for incoming JSON transaction data
        json_schema = StructType([
            StructField("record_id", IntegerType(), True),
            StructField("transacted_at", StringType(), True),
            StructField("card_number", StringType(), True),
            StructField("merchant", StringType(), True),
            StructField("category", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("first_name", StringType(), True),
            StructField("last_name", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("street", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("zip", DoubleType(), True),
            StructField("lat", DoubleType(), True),
            StructField("long", DoubleType(), True),
            StructField("city_pop", DoubleType(), True),
            StructField("job", StringType(), True),
            StructField("dob", StringType(), True),
            StructField("transaction_id", StringType(), True),
            StructField("unix_time", DoubleType(), True),
            StructField("merchant_lat", DoubleType(), True),
            StructField("merchant_long", DoubleType(), True),
            StructField("is_fraud", IntegerType(), True),
        ])

        # Create streaming DataFrame from Kafka source
        df = self.spark.readStream \
            .format("kafka") \
            .options(**kafka_options) \
            .load()

        # Parse JSON payload using defined schema
        parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), json_schema).alias("data")) \
            .select("data.*")

        parsed_df = parsed_df.withColumn(
            "transaction_time",
            coalesce(
                to_timestamp(col("transacted_at")),
                to_timestamp(from_unixtime(col("unix_time")))
            )
        )

        parsed_df = parsed_df.withColumn("amount", col("amount").cast("double")) \
            .withColumn("zip", col("zip").cast("double")) \
            .withColumn("lat", col("lat").cast("double")) \
            .withColumn("long", col("long").cast("double")) \
            .withColumn("city_pop", col("city_pop").cast("double")) \
            .withColumn("merchant_lat", col("merchant_lat").cast("double")) \
            .withColumn("merchant_long", col("merchant_long").cast("double"))

        parsed_df = parsed_df.withColumn("gender", coalesce(col("gender"), lit("unknown"))) \
            .withColumn("category", coalesce(col("category"), lit("unknown"))) \
            .withColumn("state", coalesce(col("state"), lit("unknown"))) \
            .withColumn("city", coalesce(col("city"), lit("unknown"))) \
            .withColumn("merchant", coalesce(col("merchant"), lit("unknown")))

        return parsed_df

    def add_features(self, df):
        """Add engineered features for model inference

        Args:
            df (DataFrame): Input DataFrame with raw transaction data

        Returns:
            DataFrame: DataFrame with additional feature columns
        """
        # Ensure records have usable transaction_time and amount
        df = df.dropna(subset=["transaction_time", "amount"])

        # Temporal features from transaction timestamp
        df = df.withColumn("transaction_hour", hour(col("transaction_time")))
        spark_day_of_week = dayofweek(col("transaction_time"))
        df = df.withColumn("transaction_dayofweek", ((spark_day_of_week + lit(5)) % lit(7)))
        df = df.withColumn("transaction_month", month(col("transaction_time")))
        df = df.withColumn("is_night",
                           when((col("transaction_hour") >= 22) | (col("transaction_hour") < 5), 1).otherwise(0))
        df = df.withColumn("is_weekend",
                           when(coalesce(col("transaction_dayofweek"), lit(0)) >= 5, 1).otherwise(0))
        df = df.withColumn("transaction_day", dayofmonth(col("transaction_time")))

        # Transaction pattern features (placeholders - would normally come from historical data)
        # In production, these would be calculated using window functions or join with historical data
        df = df.withColumn("time_since_last_txn", lit(0.0))  # Placeholder value
        df = df.withColumn("user_activity_24h", lit(1000))  # Placeholder value
        df = df.withColumn("rolling_avg_7d", lit(1000.0))  # Placeholder value

        # Ratio features to capture transaction amount patterns
        df = df.withColumn("amount_to_avg_ratio", col("amount") / col("rolling_avg_7d"))
        df = df.withColumn("amount_to_avg_ratio", coalesce(col("amount_to_avg_ratio"), lit(1.0)))

        # Merchant risk features from configurable list of high-risk merchants
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df = df.withColumn("merchant_risk", col("merchant").isin(high_risk_merchants).cast("int"))

        # Debug: Output schema of processed data for verification
        df.printSchema()
        return df

    def run_inference(self):
        """Main pipeline execution flow: process stream and run predictions"""
        # Local import for Spark executor compatibility
        import pandas as pd

        # Process streaming data from Kafka
        df = self.read_from_kafka()

        # Define watermark to handle late-arriving data (24 hour tolerance)
        df = df.withWatermark("transaction_time", "24 hours")

        # Add engineered features to raw data
        feature_df = self.add_features(df)

        # Get broadcasted model reference for use in UDF
        broadcast_model = self.broadcast_model

        threshold = self.prediction_threshold

        # Define prediction UDF using Pandas for vectorized operations
        @pandas_udf("struct<prediction int, fraud_probability double>")
        def predict_udf(
                amount: pd.Series,
                zip_code: pd.Series,
                lat: pd.Series,
                long_: pd.Series,
                city_pop: pd.Series,
                merchant_lat: pd.Series,
                merchant_long: pd.Series,
                transaction_hour: pd.Series,
                transaction_dayofweek: pd.Series,
                transaction_month: pd.Series,
                category: pd.Series,
                state: pd.Series,
                gender: pd.Series,
                city: pd.Series,
                merchant: pd.Series,
        ) -> pd.DataFrame:
            """Vectorized UDF returning predictions and probabilities."""
            input_df = pd.DataFrame({
                "amount": amount,
                "zip": zip_code,
                "lat": lat,
                "long": long_,
                "city_pop": city_pop,
                "merchant_lat": merchant_lat,
                "merchant_long": merchant_long,
                "transaction_hour": transaction_hour,
                "transaction_dayofweek": transaction_dayofweek,
                "transaction_month": transaction_month,
                "category": category,
                "state": state,
                "gender": gender,
                "city": city,
                "merchant": merchant,
            })

            probabilities = broadcast_model.value.predict_proba(input_df)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            return pd.DataFrame({
                "prediction": predictions,
                "fraud_probability": probabilities,
            })

        prediction_df = feature_df.withColumn(
            "prediction_struct",
            predict_udf(*[
                col(f) for f in [
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
                    "category",
                    "state",
                    "gender",
                    "city",
                    "merchant",
                ]
            ])
        )

        prediction_df = (
            prediction_df
            .withColumn("prediction", col("prediction_struct.prediction"))
            .withColumn("fraud_probability", col("prediction_struct.fraud_probability"))
            .drop("prediction_struct")
        )

        output_topic = self.config["kafka"].get("output_topic", "fraud_predictions")
        writer = (
            prediction_df.selectExpr(
                "CAST(transaction_id AS STRING) AS key",
                "to_json(struct(*)) AS value"
            )
            .writeStream
            .format("kafka")
            .option("kafka.bootstrap.servers", self.bootstrap_servers)
            .option("topic", output_topic)
            .option("kafka.security.protocol", self.security_protocol)
            .option("checkpointLocation", "checkpoints/inference")
            .outputMode("append")
        )

        if self.security_protocol.startswith("SASL"):
            writer = writer.option("kafka.sasl.mechanism", self.sasl_mechanism) \
                            .option("kafka.sasl.jaas.config", self.sasl_jaas_config)

        writer.start().awaitTermination()


if __name__ == "__main__":
    """Main entry point for the inference pipeline"""
    # Initialize pipeline with configuration
    inference = FraudDetectionInference("/app/config.yaml")

    # Start streaming processing and block until termination
    inference.run_inference()
