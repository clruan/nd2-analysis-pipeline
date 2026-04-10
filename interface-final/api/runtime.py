"""Runtime settings for the Interface-Final backend."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from .utils import ensure_directory


DEFAULT_RUNTIME_DIR = ensure_directory(Path(__file__).resolve().parent / "runtime")
DEFAULT_STATE_DB = DEFAULT_RUNTIME_DIR / "state.sqlite3"


@dataclass(frozen=True)
class RuntimeSettings:
    state_db: Path
    project_id: str
    session_id: str
    worker_heartbeat_ttl_seconds: int

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        state_db = Path(os.getenv("INTERFACE_STATE_DB", str(DEFAULT_STATE_DB))).expanduser()
        return cls(
            state_db=state_db.resolve(),
            project_id=os.getenv("INTERFACE_PROJECT_ID", "default"),
            session_id=os.getenv("INTERFACE_SESSION_ID", uuid.uuid4().hex),
            worker_heartbeat_ttl_seconds=max(1, int(os.getenv("INTERFACE_WORKER_HEARTBEAT_TTL_SECONDS", "15"))),
        )

    def with_session(self, session_id: str) -> "RuntimeSettings":
        return replace(self, session_id=session_id)
