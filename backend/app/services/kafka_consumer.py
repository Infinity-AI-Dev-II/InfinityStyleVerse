#creating a separate kafka consumer as a docker container.
#this is used to connect the TaskPulseOS.
# kafka_consumer.py
import os
import json
import logging
from confluent_kafka import Consumer
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_GROUP_ID = os.getenv('KAFKA_CONSUMER_GROUP_ID', 'skillio-consumer')
KAFKA_SECURITY_PROTOCOL = os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')

# Topic configuration
RUN_TOPICS = [
    os.getenv('KAFKA_TOPIC_RUN_STARTED', 'run.started'),
    os.getenv('KAFKA_TOPIC_RUN_UPDATED', 'run.updated'),
    os.getenv('KAFKA_TOPIC_RUN_SUCCEEDED', 'run.succeeded'),
    os.getenv('KAFKA_TOPIC_RUN_FAILED', 'run.failed'),
    os.getenv('KAFKA_TOPIC_RUN_CANCELED', 'run.canceled'),
    os.getenv('KAFKA_TOPIC_RUN_PAUSED', 'run.paused'),
    os.getenv('KAFKA_TOPIC_RUN_RESUMED', 'run.resumed'),
]

STEP_TOPICS = [
    os.getenv('KAFKA_TOPIC_STEP_STARTED', 'step.started'),
    os.getenv('KAFKA_TOPIC_STEP_SUCCEEDED', 'step.succeeded'),
    os.getenv('KAFKA_TOPIC_STEP_FAILED', 'step.failed'),
    os.getenv('KAFKA_TOPIC_STEP_RETRYING', 'step.retrying'),
    os.getenv('KAFKA_TOPIC_STEP_COMPENSATING', 'step.compensating'),
]

ALERT_TOPICS = [
    os.getenv('KAFKA_TOPIC_ALERT_RAISED', 'alert.raised'),
    os.getenv('KAFKA_TOPIC_ALERT_ACKNOWLEDGED', 'alert.acknowledged'),
    os.getenv('KAFKA_TOPIC_ALERT_CLEARED', 'alert.cleared'),
]

ALL_TOPICS = RUN_TOPICS + STEP_TOPICS + ALERT_TOPICS


# ─────────────────────────────────────────────────────────
# MESSAGE HANDLERS
# ─────────────────────────────────────────────────────────

def handle_run_message(msg_data: dict, topic: str):
    """Handle run lifecycle events"""
    logger.info(f"[RUN] {topic}: {msg_data}")
    # TODO: Connect to TaskPulseOS


def handle_step_message(msg_data: dict, topic: str):
    """Handle step execution events"""
    logger.info(f"[STEP] {topic}: {msg_data}")
    # TODO: Connect to TaskPulseOS


def handle_alert_message(msg_data: dict, topic: str):
    """Handle alert events"""
    logger.info(f"[ALERT] {topic}: {msg_data}")
    # TODO: Connect to TaskPulseOS


# ─────────────────────────────────────────────────────────
# MESSAGE ROUTING
# ─────────────────────────────────────────────────────────

def route_message(topic: str, msg_data: dict):
    """Route message to appropriate handler based on topic"""
    if topic in RUN_TOPICS:
        handle_run_message(msg_data, topic)
    elif topic in STEP_TOPICS:
        handle_step_message(msg_data, topic)
    elif topic in ALERT_TOPICS:
        handle_alert_message(msg_data, topic)
    else:
        logger.warning(f"Unknown topic: {topic}")


def process_message(msg):
    """Process Kafka message and deserialize JSON"""
    try:
        topic = msg.topic()
        value = msg.value().decode('utf-8')
        msg_data = json.loads(value)
        
        # Route to appropriate handler
        route_message(topic, msg_data)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")


# ─────────────────────────────────────────────────────────
# KAFKA CONSUMER SETUP
# ─────────────────────────────────────────────────────────

def create_consumer():
    """Create and configure Kafka consumer"""
    config = {
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'security.protocol': KAFKA_SECURITY_PROTOCOL,
    }
    
    return Consumer(config)


def start_consumer():
    """Start consuming messages from Kafka"""
    consumer = create_consumer()
    
    logger.info(f"Subscribing to topics: {ALL_TOPICS}")
    consumer.subscribe(ALL_TOPICS)
    
    logger.info("Starting Kafka consumer...")
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue
            
            # Process message
            process_message(msg)
            
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    finally:
        logger.info("Closing consumer...")
        consumer.close()


if __name__ == "__main__":
    start_consumer()
