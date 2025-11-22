from datetime import datetime
import uuid
from ..database import db

class RequestLog(db.Model):
    __tablename__ = "request_log"
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), nullable=True)
    user_id = db.Column(db.Integer, nullable=True)
    endpoint = db.Column(db.String(255), nullable=False)
    method = db.Column(db.String(10), nullable=False)
    status_code = db.Column(db.Integer, nullable=True)
    latency_ms = db.Column(db.Float, nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class IdempotencyKey(db.Model):
    __tablename__ = "idempotency_keys"
    key = db.Column(db.String(128), primary_key=True)
    request_hash = db.Column(db.String(128), nullable=True)
    response_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Decision(db.Model):
    __tablename__ = "decisions"
    request_id = db.Column(db.String(128), primary_key=True)
    policy_id = db.Column(db.String(128), nullable=False)
    task = db.Column(db.String(128), nullable=False)
    tenant = db.Column(db.String(128), nullable=True)
    ab_bucket = db.Column(db.String(32), nullable=True)
    context_json = db.Column(db.JSON, nullable=True)
    candidates_json = db.Column(db.JSON, nullable=True)
    chosen_json = db.Column(db.JSON, nullable=True)
    scores_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Event(db.Model):
    __tablename__ = "events"
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    request_id = db.Column(db.String(128), db.ForeignKey("decisions.request_id"), nullable=True, index=True)
    type = db.Column(db.String(64), nullable=False)
    value = db.Column(db.Float, nullable=True)
    meta_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Reward(db.Model):
    __tablename__ = "rewards"
    request_id = db.Column(db.String(128), db.ForeignKey("decisions.request_id"), primary_key=True)
    reward = db.Column(db.Float, nullable=False)
    components_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Policy(db.Model):
    __tablename__ = "policies"
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    task = db.Column(db.String(128), nullable=False, index=True)
    version = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(32), default="staged")
    weights_blob = db.Column(db.LargeBinary, nullable=True)
    config_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Experiment(db.Model):
    __tablename__ = "experiments"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(128), nullable=False)
    task = db.Column(db.String(128), nullable=True)
    design_json = db.Column(db.JSON, nullable=True)
    status = db.Column(db.String(32), default="draft")
    started_at = db.Column(db.DateTime, nullable=True)
    stopped_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
