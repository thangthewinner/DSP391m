#!/bin/bash

# Quick setup script for AI Proctoring System

set -e

echo "=========================================="
echo "AI Proctoring System - Quick Setup"
echo "=========================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download models
echo ""
echo "[5/5] Downloading models..."
python scripts/download_models.py

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Copy .env.example to .env: cp .env.example .env"
echo "  3. Run server: uvicorn src.api.main:app --reload"
echo ""
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
