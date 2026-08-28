@echo off
echo =========================================
echo Kipsigis LLM - Inference Engine
echo =========================================

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run run_train.bat first!
    pause
    exit /b
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [INFO] Starting inference...
python inference.py

echo.
echo [INFO] Inference finished.
pause
