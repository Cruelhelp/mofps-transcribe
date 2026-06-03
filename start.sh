#!/bin/sh
set -eu

if [ "${STREAMLIT_SERVER_PORT:-}" = '$PORT' ]; then
  unset STREAMLIT_SERVER_PORT
fi

python -m streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-7860}" \
  --server.enableCORS false \
  --server.enableXsrfProtection false
