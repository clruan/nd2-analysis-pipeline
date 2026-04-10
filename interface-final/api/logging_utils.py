"""Structured logging helpers for the Interface-Final backend."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format="%(message)s")
    root_logger.setLevel(level)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level),
        "logger": logger.name,
        "event": event,
    }
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, datetime):
            payload[key] = value.astimezone(timezone.utc).isoformat()
        else:
            payload[key] = value
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))
