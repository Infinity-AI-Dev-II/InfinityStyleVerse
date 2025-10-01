# backend/app/models/WaitStepTimer.py
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from ..database import Base

class WaitStepTimer(Base):
    __tablename__ = "wait_step_timers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, nullable=False)
    step_id = Column(String, nullable=False)
    trigger_at = Column(DateTime, nullable=False)  # when timer expires
    payload_json = Column(JSON, nullable=True)
    status = Column(String, default="pending")  # pending, triggered
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
