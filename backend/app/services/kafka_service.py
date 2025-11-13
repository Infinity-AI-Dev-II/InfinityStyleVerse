import json
from flask import current_app

try:
    from confluent_kafka import Producer
except Exception:
    Producer = None

def get_kafka_producer():
    if Producer is None:
        return None
    servers = current_app.config.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    try:
        return Producer({"bootstrap.servers": servers})
    except Exception as e:
        current_app.logger.warning("Kafka producer init failed: %s", e)
        return None

def publish_kafka(topic: str, payload: dict):
    p = get_kafka_producer()
    if p is None:
        current_app.logger.debug("Kafka not available; skipping publish to %s", topic)
        return False
    try:
        p.produce(topic, json.dumps(payload).encode("utf-8"))
        p.flush(timeout=1.0)
        return True
    except Exception as e:
        current_app.logger.warning("Kafka publish failed to topic %s: %s", topic, e)
        return False
