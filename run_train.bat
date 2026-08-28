@echo off
echo =========================================
echo Kipsigis LLM - Training Pipeline
echo =========================================

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing/Updating dependencies...
pip install -r requirements.txt

echo.
echo [INFO] Starting training process...
python train.py

echo.
echo [INFO] Training finished.
pause
