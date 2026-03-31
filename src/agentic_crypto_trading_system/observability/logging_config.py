"""Structured JSON logging configuration.

Provides correlation IDs for request tracing and structured output
for log aggregation systems.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure structured logging for the application."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    root.handlers = [handler]


class StructuredLogger:
    """Logger with correlation ID support."""

    def __init__(self, name: str, correlation_id: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]

    def _log(self, level: int, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, msg, (), None
        )
        record.correlation_id = self.correlation_id
        if data:
            record.extra_data = data
        self.logger.handle(record)

    def info(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.INFO, msg, data)

    def warning(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.WARNING, msg, data)

    def error(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.ERROR, msg, data)

    def debug(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.DEBUG, msg, data)
