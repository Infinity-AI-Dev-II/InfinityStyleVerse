# backend/app/models/runs.py
from datetime import datetime
from ..database import db

class Run(db.Model):
    __tablename__ = "runs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    workflow_id = db.Column(db.Integer, db.ForeignKey("workflow_defs.id"), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    tenant = db.Column(db.String(100))
    caller = db.Column(db.String(100))
    inputs_json = db.Column(db.Text)
    
    # --- new columns ---
    started_at = db.Column(db.DateTime)  # timestamp when run starts
    ended_at = db.Column(db.DateTime)    # timestamp when run ends
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    steps = db.relationship("RunStep", backref="run", lazy=True)
    vars = db.relationship("RunVar", backref="run", lazy=True)
    signals = db.relationship("Signal", backref="run", lazy=True)
    locks = db.relationship("Lock", backref="run", lazy=True)
    compensations = db.relationship("Compensation", backref="run", lazy=True)
