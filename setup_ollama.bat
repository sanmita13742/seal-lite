@echo off
REM Quick start script for Windows/PowerShell users

echo ========================================
echo SEAL with Ollama - Quick Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ first.
    exit /b 1
)

echo [1/4] Checking Python... OK
echo.

REM Check Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama not running!
    echo.
    echo Please:
    echo   1. Install Ollama from https://ollama.ai
    echo   2. Run: ollama serve
    echo   3. Run: ollama pull llama3.2:latest
    echo.
    exit /b 1
)

echo [2/4] Checking Ollama... OK
echo.

REM Install dependencies
echo [3/4] Installing dependencies...
pip install -q -r requirements_ollama.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo       Dependencies installed OK
echo.

REM Run connectivity check
echo [4/4] Testing Ollama connection...
python run_ollama_simple.py --mode=check
if errorlevel 1 (
    echo ERROR: Connection test failed
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run ARC evaluation (3 tasks):
echo      python run_ollama_simple.py --mode=arc --num_examples=3
echo.
echo   2. Run knowledge test:
echo      python run_ollama_simple.py --mode=knowledge
echo.
echo   3. Read detailed guide:
echo      type README_OLLAMA.md
echo.
echo ========================================
