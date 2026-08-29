#!/usr/bin/env bash
set -e

echo "========================================="
echo "Kipsigis LLM - Training Pipeline"
echo "========================================="

if [ ! -d "venv" ]; then
    echo "[INFO] Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Installing/Updating dependencies..."
pip install -r requirements.txt

echo ""
echo "[INFO] Starting training process..."
python train.py "$@"

echo ""
echo "[INFO] Training finished."
