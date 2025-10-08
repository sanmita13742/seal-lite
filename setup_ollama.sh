#!/bin/bash
# Quick start script for Linux/Mac users

echo "========================================"
echo "SEAL with Ollama - Quick Setup"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found!"
    echo "Please install Python 3.10+ first."
    exit 1
fi

echo "[1/4] Checking Python... OK"
echo

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama not running!"
    echo
    echo "Please:"
    echo "  1. Install Ollama from https://ollama.ai"
    echo "  2. Run: ollama serve"
    echo "  3. Run: ollama pull llama3.2:latest"
    echo
    exit 1
fi

echo "[2/4] Checking Ollama... OK"
echo

# Install dependencies
echo "[3/4] Installing dependencies..."
pip install -q -r requirements_ollama.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "      Dependencies installed OK"
echo

# Run connectivity check
echo "[4/4] Testing Ollama connection..."
python3 run_ollama_simple.py --mode=check
if [ $? -ne 0 ]; then
    echo "ERROR: Connection test failed"
    exit 1
fi

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "  1. Run ARC evaluation (3 tasks):"
echo "     python3 run_ollama_simple.py --mode=arc --num_examples=3"
echo
echo "  2. Run knowledge test:"
echo "     python3 run_ollama_simple.py --mode=knowledge"
echo
echo "  3. Read detailed guide:"
echo "     cat README_OLLAMA.md"
echo
echo "========================================"
