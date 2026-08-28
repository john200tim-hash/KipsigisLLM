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
echo [INFO] Please make sure your Hugging Face token is set.
echo You can set it temporarily in this window using: set HF_TOKEN=your_token
echo.

python src/download_asr.py

echo.
pause
