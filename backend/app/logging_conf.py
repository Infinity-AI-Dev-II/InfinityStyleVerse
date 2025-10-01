# backend/app/logging_conf.py
import logging
import sys
import json
from typing import Any

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # include extras if present (task_id, request_id, etc.)
        for k in ("task_id", "request_id", "workflow_id", "run_id", "step_id", "tenant", "trace_id"):
            if hasattr(record, k):
              base[k] = getattr(record, k)

        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base)

def configure_logging(level: int | str = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]
    root.addHandler(handler)
    root.setLevel(level)

