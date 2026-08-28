@echo off
echo =========================================
echo Kipsigis LLM - ASR Dataset Downloader
echo =========================================

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run run_train.bat first to set it up!
    pause
    exit /b
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Updating dependencies...
pip install -r requirements.txt

echo.
IF "%HF_TOKEN%"=="" (
    set /p HF_TOKEN="Enter your Hugging Face Token (starts with hf_): "
)
echo.

python src/download_asr.py

echo.
pause
