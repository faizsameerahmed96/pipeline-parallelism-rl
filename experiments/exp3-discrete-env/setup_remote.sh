#!/bin/bash
# Setup script for running experiments on a remote GPU machine
# Usage: bash setup_remote.sh

set -e

echo "=== Installing system dependencies ==="
apt-get update
apt-get install -y swig build-essential

echo "=== Installing Python dependencies ==="
pip3 install --break-system-packages \
    "tensorboard>=2.10.0" \
    "wandb>=0.13.11" \
    "gym==0.26.2" \
    "gymnasium[box2d]==1.2.0" \
    "mujoco==3.3.6" \
    "moviepy>=1.0.3" \
    "pygame>=2.1" \
    "huggingface-hub>=0.11.1" \
    "rich<12.0" \
    "tenacity>=8.2.2" \
    "tyro>=0.5.10" \
    "torchviz>=0.0.3" \
    "torchinfo>=1.8.0" \
    "psutil>=5.9.0"

echo "=== Verifying GPU ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"

echo "=== Setup complete ==="
