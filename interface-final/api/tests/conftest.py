"""Shared test setup for Interface-Final backend tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path


if "joblib" not in sys.modules:
    joblib_stub = types.ModuleType("joblib")
    joblib_stub.Parallel = lambda *args, **kwargs: None  # type: ignore[assignment]
    joblib_stub.delayed = lambda func, *args, **kwargs: func  # type: ignore[assignment]
    sys.modules["joblib"] = joblib_stub

if "pyclesperanto" not in sys.modules:
    sys.modules["pyclesperanto"] = types.ModuleType("pyclesperanto")

if "nd2reader" not in sys.modules:
    nd2_stub = types.ModuleType("nd2reader")
    nd2_stub.ND2Reader = object  # type: ignore[attr-defined]
    nd2_stub.Nd2 = object  # type: ignore[attr-defined]
    sys.modules["nd2reader"] = nd2_stub


INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))
