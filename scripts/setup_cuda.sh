#!/bin/bash
# Setup script for CUDA environment

set -e

echo "=========================================="
echo "Setting up DSP391m with CUDA support"
echo "=========================================="
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found!"
    echo "   Make sure NVIDIA drivers are installed."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show CUDA info
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA Device Info:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
fi

# Detect CUDA version
CUDA_VERSION="cu126"  # Default to CUDA 12.6
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "NVIDIA Driver: $DRIVER_VERSION"
    
    # Suggest CUDA version based on driver
    if [[ $(echo "$DRIVER_VERSION >= 525.60" | bc -l) -eq 1 ]]; then
        CUDA_VERSION="cu126"  # CUDA 12.6
        echo "Recommended: CUDA 12.6"
    elif [[ $(echo "$DRIVER_VERSION >= 450.80" | bc -l) -eq 1 ]]; then
        CUDA_VERSION="cu118"  # CUDA 11.8
        echo "Recommended: CUDA 11.8"
    fi
    echo ""
fi

# Install base dependencies
echo "[1/4] Installing base dependencies..."
uv pip install -r requirements.txt --exclude torch --exclude torchvision --exclude torchaudio
echo "✓ Base dependencies installed"
echo ""

# Install PyTorch with CUDA
echo "[2/4] Installing PyTorch with CUDA $CUDA_VERSION..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_VERSION
echo "✓ PyTorch with CUDA installed"
echo ""

# Verify CUDA
echo "[3/4] Verifying CUDA installation..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CUDA not available - will use CPU")
EOF
echo ""

# Update .env
echo "[4/4] Updating .env for CUDA..."
if [ -f .env ]; then
    # Backup
    cp .env .env.backup
    
    # Update TORCH_DEVICE
    if grep -q "TORCH_DEVICE=" .env; then
        sed -i 's/TORCH_DEVICE=.*/TORCH_DEVICE=cuda/' .env
    else
        echo "TORCH_DEVICE=cuda" >> .env
    fi
    
    # Add CUDA_VISIBLE_DEVICES if not exists
    if ! grep -q "CUDA_VISIBLE_DEVICES=" .env; then
        echo "CUDA_VISIBLE_DEVICES=0" >> .env
    fi
    
    echo "✓ .env updated (backup saved to .env.backup)"
else
    echo "⚠️  .env not found - creating from .env.example"
    cp .env.example .env
    sed -i 's/TORCH_DEVICE=.*/TORCH_DEVICE=cuda/' .env
fi
echo ""

# Reconvert model for GPU (optional)
echo "=========================================="
echo "Optional: Reconvert model for GPU"
echo "=========================================="
echo ""
echo "For better GPU performance, reconvert PhoWhisper with float16:"
echo ""
echo "  uv run ct2-transformers-converter \\"
echo "    --model models/stt/phowhisper-small \\"
echo "    --output_dir models/stt/phowhisper-small-ct2-gpu \\"
echo "    --quantization float16"
echo ""
echo "Then update .env:"
echo "  STT_MODEL_PATH=models/stt/phowhisper-small-ct2-gpu"
echo ""

echo "=========================================="
echo "✅ CUDA setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start server: uv run uvicorn src.api.main:app --reload"
echo "  2. Check GPU usage: nvidia-smi"
echo ""
