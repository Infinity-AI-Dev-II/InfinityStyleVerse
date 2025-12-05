from ..database import db
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime


class Heartbeat(db.Model):
    __tablename__ = "heartbeats"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    worker_id = db.Column(db.String(128), nullable=False, index=True)
    kind = db.Column(db.String(64), nullable=False)
    last_seen_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    meta_json = db.Column(JSONB, nullable=True)

    def __repr__(self):
        return f"<Heartbeat worker_id={self.worker_id} kind={self.kind} last_seen_at={self.last_seen_at}>"



