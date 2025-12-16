import json
import os
from flask import current_app
import logging
from kafka.admin import KafkaAdminClient, NewTopic

try:
    from confluent_kafka import Producer
except Exception:
    Producer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
def create_kafka_topics():
    try:
            bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
            topics = [
            "run.started", "run.updated", "run.succeeded", "run.failed",
            "run.canceled", "run.paused", "run.resumed", "step.started",
            "step.succeeded", "step.failed", "step.retrying", "step.compensating",
            "alert.raised", "alert.acknowledged", "alert.cleared"
            ]
            admin_client = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                client_id='topic-creator'
            )
            # Create topic objects
            topic_list = [NewTopic(name=topic, num_partitions=3, replication_factor=1) 
                        for topic in topics]
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            logger.info("Kafka topics created successfully.")
            existing_topics = admin_client.list_topics()
            logger.info(f"Available topics: {existing_topics}")
            admin_client.close()
    except Exception as e:        
            logger.error(f"Failed to create Kafka topics: {e}")