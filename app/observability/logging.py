"""Single-line JSON logging with automatic trace_id injection and PII masking."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.observability.sanitizer import sanitize

_FIELDS_ATTR = "rag_fields"


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={_FIELDS_ATTR: fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Imported lazily to avoid a circular import with trace.py.
        from app.observability.trace import current_trace_id

        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "trace_id": current_trace_id(),
        }
        payload.update(getattr(record, _FIELDS_ATTR, {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(sanitize(payload), ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
