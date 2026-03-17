#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_VENV_DIR="$ROOT_DIR/interface-final/api/venv"
API_REQUIREMENTS="$ROOT_DIR/interface-final/api/requirements.txt"
WEB_DIR="$ROOT_DIR/interface-final/web"
API_PORT="${API_PORT:-8100}"
WEB_PORT="${WEB_PORT:-5173}"
VITE_API_PROXY_TARGET="${VITE_API_PROXY_TARGET:-http://127.0.0.1:${API_PORT}}"
export VITE_API_PROXY_TARGET WEB_PORT

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

require_cmd python3
require_cmd npm

if [[ ! -d "$API_VENV_DIR" ]]; then
  echo "Creating backend virtualenv..."
  python3 -m venv "$API_VENV_DIR"
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

echo "Starting backend at http://127.0.0.1:${API_PORT}"
"$API_VENV_DIR/bin/uvicorn" --app-dir "$ROOT_DIR/interface-final" api.main:app --port "$API_PORT" &
API_PID=$!

echo "Starting frontend at http://127.0.0.1:${WEB_PORT}"
echo "Frontend API proxy target: ${VITE_API_PROXY_TARGET}"
npm --prefix "$WEB_DIR" run dev -- --host 127.0.0.1 --port "$WEB_PORT" &
WEB_PID=$!

cleanup() {
  local exit_code=$?
  if kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  if kill -0 "$WEB_PID" >/dev/null 2>&1; then
    kill "$WEB_PID" >/dev/null 2>&1 || true
  fi
  wait "$API_PID" >/dev/null 2>&1 || true
  wait "$WEB_PID" >/dev/null 2>&1 || true
  exit "$exit_code"
}

trap cleanup INT TERM EXIT

while true; do
  if ! kill -0 "$API_PID" >/dev/null 2>&1; then
    wait "$API_PID" || true
    break
  fi
  if ! kill -0 "$WEB_PID" >/dev/null 2>&1; then
    wait "$WEB_PID" || true
    break
  fi
  sleep 1
done
