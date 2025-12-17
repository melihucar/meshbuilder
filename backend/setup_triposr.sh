#!/bin/bash

# TripoSR Setup Script
# Downloads TripoSR repository and model weights
# Note: torchmcubes is replaced by our PyMCubes compatibility shim

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRIPOSR_DIR="$SCRIPT_DIR/triposr"

echo "Setting up TripoSR for high-quality 3D mesh generation..."
echo ""

# Check if already cloned
if [ -d "$TRIPOSR_DIR" ]; then
    echo "TripoSR directory exists. Updating..."
    cd "$TRIPOSR_DIR"
    git pull || true
else
    echo "Cloning TripoSR repository..."
    git clone https://github.com/VAST-AI-Research/TripoSR.git "$TRIPOSR_DIR"
fi

cd "$SCRIPT_DIR"

echo ""
echo "Syncing dependencies..."
uv sync

echo ""
echo "Downloading model weights (this may take a while - ~1.5GB)..."

# Create pretrained directory
mkdir -p "$TRIPOSR_DIR/pretrained"

# Download model weights from Hugging Face using Python
uv run python3 -c "
from huggingface_hub import hf_hub_download
import os

model_dir = '$TRIPOSR_DIR/pretrained'

print('Downloading TripoSR model from Hugging Face...')

# Download model checkpoint
ckpt_path = hf_hub_download(
    repo_id='stabilityai/TripoSR',
    filename='model.ckpt',
    local_dir=model_dir,
    local_dir_use_symlinks=False
)
print(f'Downloaded: {ckpt_path}')

# Download config
config_path = hf_hub_download(
    repo_id='stabilityai/TripoSR',
    filename='config.yaml',
    local_dir=model_dir,
    local_dir_use_symlinks=False
)
print(f'Downloaded: {config_path}')

print('Model weights downloaded successfully!')
"

echo ""
echo "============================================"
echo "TripoSR setup complete!"
echo ""
echo "Note: We use PyMCubes instead of torchmcubes"
echo "(no C++ compilation required)"
echo ""
echo "Restart the backend to load the model:"
echo "  ./dev restart"
echo "============================================"
