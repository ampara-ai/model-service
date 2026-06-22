#!/bin/bash
set -euo pipefail

echo "=== AMPARA Model Service ==="
echo "MODELS_DIR      : ${MODELS_DIR:-/app/models}"
echo "MODEL_PRECISION : ${MODEL_PRECISION:-int8}"
echo ""
echo "--- Verificando / descargando modelos ---"
python download_models.py

echo ""
echo "--- Iniciando uvicorn ---"
exec uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8001}" \
    --workers "${UVICORN_WORKERS:-1}" \
    --log-level info
