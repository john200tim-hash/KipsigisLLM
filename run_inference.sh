#!/usr/bin/env bash
set -e

echo "========================================="
echo "Kipsigis LLM - Inference Engine"
echo "========================================="

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found. Please run run_train.sh first!"
    exit 1
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[INFO] Starting inference..."
python inference.py "$@"

echo ""
echo "[INFO] Inference finished."
