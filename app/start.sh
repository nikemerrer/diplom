#!/bin/sh
set -e

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8000}"
PUBLIC_URL="${PUBLIC_URL:-http://localhost:${PORT}}"

echo "-------------------------------------------------------------"
echo "Web UI:     ${PUBLIC_URL}"
echo "Healthcheck: ${PUBLIC_URL%/}/health"
echo "-------------------------------------------------------------"

exec uvicorn main:app --host "${HOST}" --port "${PORT}"
