# backend/app/models/workflow_defs.py
from datetime import datetime
from ..database import db
import json

class WorkflowDef(db.Model):
    __tablename__ = "workflow_defs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    dsl_yaml = db.Column(db.Text, nullable=False)      
    dag_json = db.Column(db.Text, nullable=True)       
    created_by = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    runs = db.relationship("Run", backref="workflow_def", lazy=True)

    def to_dict(self, include_dag=False):
        base = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "dsl_yaml": self.dsl_yaml,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_dag and self.dag_json:
            try:
                base["dag"] = json.loads(self.dag_json)
            except Exception:
                base["dag"] = None
        return base
