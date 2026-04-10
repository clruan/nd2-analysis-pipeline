#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_REQUIREMENTS="$ROOT_DIR/interface-final/api/requirements.txt"
API_RUNTIME_DIR="$ROOT_DIR/interface-final/api/runtime"
WEB_DIR="$ROOT_DIR/interface-final/web"
API_PORT="${API_PORT:-8100}"
WEB_PORT="${WEB_PORT:-5173}"
VITE_API_PROXY_TARGET="${VITE_API_PROXY_TARGET:-http://127.0.0.1:${API_PORT}}"
INTERFACE_CACHE_DIR="${INTERFACE_CACHE_DIR:-$API_RUNTIME_DIR/cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$INTERFACE_CACHE_DIR/matplotlib}"
MICROSCOPY_RUNTIME_CACHE_DIR="${MICROSCOPY_RUNTIME_CACHE_DIR:-$INTERFACE_CACHE_DIR/microscopy}"
SCYJAVA_CACHE_DIR="${SCYJAVA_CACHE_DIR:-$MICROSCOPY_RUNTIME_CACHE_DIR/scyjava}"
SCYJAVA_M2_REPO="${SCYJAVA_M2_REPO:-$MICROSCOPY_RUNTIME_CACHE_DIR/m2}"
SCYJAVA_JAVA_VENDOR="${SCYJAVA_JAVA_VENDOR:-zulu}"
SCYJAVA_JAVA_VERSION="${SCYJAVA_JAVA_VERSION:-11}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$INTERFACE_CACHE_DIR/xdg}"
CJDK_CACHE_DIR="${CJDK_CACHE_DIR:-$MICROSCOPY_RUNTIME_CACHE_DIR/cjdk}"
export VITE_API_PROXY_TARGET WEB_PORT
export PYTHONPATH="$ROOT_DIR/interface-final${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR MICROSCOPY_RUNTIME_CACHE_DIR SCYJAVA_CACHE_DIR SCYJAVA_M2_REPO SCYJAVA_JAVA_VENDOR SCYJAVA_JAVA_VERSION XDG_CACHE_HOME CJDK_CACHE_DIR

mkdir -p "$API_RUNTIME_DIR" "$MPLCONFIGDIR" "$SCYJAVA_CACHE_DIR" "$SCYJAVA_M2_REPO" "$XDG_CACHE_HOME" "$CJDK_CACHE_DIR"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

python_version() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

python_supported_for_interface() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info < (3, 13) else 1)
PY
}

select_api_python() {
  local candidate
  for candidate in "${API_PYTHON_BIN:-}" python3.12 python3.11 python3; do
    [[ -n "$candidate" ]] || continue
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if python_supported_for_interface "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

require_cmd npm

if ! API_PYTHON_BIN="$(select_api_python)"; then
  echo "Interface-Final requires Python 3.12 or 3.11 for microscopy format support." >&2
  echo "The current default Python 3.13 cannot install the aicsimageio dependency used for .czi/.oib/.oif files." >&2
  echo "Install python3.12 or python3.11, or set API_PYTHON_BIN to a compatible interpreter." >&2
  exit 1
fi

API_PYTHON_VERSION="$(python_version "$API_PYTHON_BIN")"
API_VENV_DIR="${API_VENV_DIR:-$ROOT_DIR/interface-final/api/venv-py${API_PYTHON_VERSION}}"

if [[ ! -d "$API_VENV_DIR" ]]; then
  echo "Creating backend virtualenv with $API_PYTHON_BIN ($API_PYTHON_VERSION)..."
  "$API_PYTHON_BIN" -m venv "$API_VENV_DIR"
fi

if [[ ! -x "$API_VENV_DIR/bin/python" ]]; then
  echo "Backend virtualenv is missing python executable at $API_VENV_DIR/bin/python" >&2
  exit 1
fi

if [[ ! -d "$WEB_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm --prefix "$WEB_DIR" install
fi

DEPS_STAMP="$API_VENV_DIR/.deps_installed"
if [[ ! -f "$DEPS_STAMP" || "$API_REQUIREMENTS" -nt "$DEPS_STAMP" ]]; then
  echo "Installing backend dependencies..."
  if "$API_VENV_DIR/bin/pip" install -r "$API_REQUIREMENTS"; then
    touch "$DEPS_STAMP"
  else
    echo "Warning: backend dependency install failed; continuing with existing venv packages."
    echo "If startup fails, run: $API_VENV_DIR/bin/pip install -r $API_REQUIREMENTS"
  fi
fi

echo "Using backend Python: $API_PYTHON_BIN ($API_PYTHON_VERSION)"
echo "Using backend virtualenv: $API_VENV_DIR"

echo "Starting backend at http://127.0.0.1:${API_PORT}"
"$API_VENV_DIR/bin/uvicorn" --app-dir "$ROOT_DIR/interface-final" api.main:app --port "$API_PORT" &
API_PID=$!

echo "Starting worker"
"$API_VENV_DIR/bin/python" -m api.worker &
WORKER_PID=$!

echo "Starting frontend at http://127.0.0.1:${WEB_PORT}"
echo "Frontend API proxy target: ${VITE_API_PROXY_TARGET}"
npm --prefix "$WEB_DIR" run dev -- --host 127.0.0.1 --port "$WEB_PORT" &
WEB_PID=$!

cleanup() {
  local exit_code=$?
  if kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  if kill -0 "$WORKER_PID" >/dev/null 2>&1; then
    kill "$WORKER_PID" >/dev/null 2>&1 || true
  fi
  if kill -0 "$WEB_PID" >/dev/null 2>&1; then
    kill "$WEB_PID" >/dev/null 2>&1 || true
  fi
  wait "$API_PID" >/dev/null 2>&1 || true
  wait "$WORKER_PID" >/dev/null 2>&1 || true
  wait "$WEB_PID" >/dev/null 2>&1 || true
  exit "$exit_code"
}

trap cleanup INT TERM EXIT

while true; do
  if ! kill -0 "$API_PID" >/dev/null 2>&1; then
    wait "$API_PID" || true
    break
  fi
  if ! kill -0 "$WORKER_PID" >/dev/null 2>&1; then
    wait "$WORKER_PID" || true
    break
  fi
  if ! kill -0 "$WEB_PID" >/dev/null 2>&1; then
    wait "$WEB_PID" || true
    break
  fi
  sleep 1
done
