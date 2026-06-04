#!/bin/sh
set -eu

python -m uvicorn web_app:app \
  --host 0.0.0.0 \
  --port "${PORT:-7860}" \
  --workers 1
