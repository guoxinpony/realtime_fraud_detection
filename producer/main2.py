import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, Optional

from confluent_kafka import Producer
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="../.env")


def build_producer() -> Producer:
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    username = os.getenv("KAFKA_USERNAME")
    password = os.getenv("KAFKA_PASSWORD")

    config: Dict[str, Any] = {
        "bootstrap.servers": bootstrap_servers,
        "client.id": "transaction-producer-dataset",
        "compression.type": "gzip",
        "linger.ms": 5,
        "batch.size": 16384,
    }

    if username and password:
        config.update(
            {
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": username,
                "sasl.password": password,
            }
        )
    else:
        config["security.protocol"] = "PLAINTEXT"

    try:
        producer = Producer(config)
        logger.info("Kafka producer ready (bootstrap=%s)", bootstrap_servers)
        return producer
    except Exception as exc:  # pragma: no cover - fatal startup error
        logger.error("Failed to construct Kafka producer: %s", exc)
        raise


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        logger.debug("Unable to parse float from %s", value)
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        logger.debug("Unable to parse int from %s", value)
        return None


class DatasetProducer:
    def __init__(self, file_path: str, topic: str, sleep_interval: float, limit: Optional[int]):
        self.file_path = file_path
        self.topic = topic
        self.sleep_interval = sleep_interval
        self.limit = limit
        self.producer = build_producer()
        self.running = False
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _transform_row(self, row: Dict[str, str], index: int) -> Optional[Dict[str, Any]]:
        transaction_id = row.get("trans_num") or row.get("transaction_id")
        if not transaction_id:
            logger.warning("Skipping row %s without transaction id", index)
            return None

        payload: Dict[str, Any] = {
            "record_id": parse_int(row.get("")) or index,
            "transacted_at": row.get("trans_date_trans_time"),
            "card_number": row.get("cc_num"),
            "merchant": row.get("merchant"),
            "category": row.get("category"),
            "amount": parse_float(row.get("amt")),
            "first_name": row.get("first"),
            "last_name": row.get("last"),
            "gender": row.get("gender"),
            "street": row.get("street"),
            "city": row.get("city"),
            "state": row.get("state"),
            "zip": parse_int(row.get("zip")),
            "lat": parse_float(row.get("lat")),
            "long": parse_float(row.get("long")),
            "city_pop": parse_int(row.get("city_pop")),
            "job": row.get("job"),
            "dob": row.get("dob"),
            "transaction_id": transaction_id,
            "unix_time": parse_int(row.get("unix_time")),
            "merchant_lat": parse_float(row.get("merch_lat")),
            "merchant_long": parse_float(row.get("merch_long")),
            "is_fraud": parse_int(row.get("is_fraud")),
        }
        return payload

    def _publish(self, record: Dict[str, Any]) -> None:
        try:
            self.producer.produce(
                self.topic,
                key=str(record["transaction_id"]),
                value=json.dumps(record),
                on_delivery=lambda err, msg: logger.error("Delivery failed: %s", err)
                if err
                else None,
            )
            self.producer.poll(0)
        except BufferError:
            self.producer.poll(5)
            self.producer.produce(
                self.topic,
                key=str(record["transaction_id"]),
                value=json.dumps(record),
            )
        except Exception as exc:
            logger.error("Failed to publish record %s: %s", record.get("transaction_id"), exc)

    def run(self) -> None:
        if not os.path.exists(self.file_path):
            logger.error("CSV file not found: %s", self.file_path)
            sys.exit(1)

        self.running = True
        sent = 0
        logger.info("Starting dataset streaming from %s", self.file_path)

        with open(self.file_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for index, row in enumerate(reader):
                if not self.running:
                    break

                record = self._transform_row(row, index)
                if not record:
                    continue

                self._publish(record)
                sent += 1

                if self.sleep_interval > 0:
                    time.sleep(self.sleep_interval)

                if self.limit and sent >= self.limit:
                    logger.info("Reached message limit: %s", self.limit)
                    break

        self._shutdown()
        logger.info("Dataset streaming completed. Messages sent: %s", sent)

    def _shutdown(self, signum=None, frame=None) -> None:
        if self.running:
            logger.info("Shutting down dataset producer")
            self.running = False
        if hasattr(self, "producer") and self.producer is not None:
            self.producer.flush(timeout=30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send real transactions to Kafka")
    parser.add_argument(
        "--file",
        default="../data/fraudTrain.csv",
        help="CSV file with training transactions",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC", "transactions"),
        help="Kafka topic to publish to",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Delay between records in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of records to send",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    producer = DatasetProducer(
        file_path=args.file,
        topic=args.topic,
        sleep_interval=args.sleep,
        limit=args.limit,
    )
    try:
        producer.run()
    except KeyboardInterrupt:
        producer._shutdown()
