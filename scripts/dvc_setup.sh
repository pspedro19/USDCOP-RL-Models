#!/bin/bash
#
# DVC Setup Script
# ================
#
# Initializes DVC for data versioning in the USDCOP RL Trading project.
#
# Prerequisites:
#   - Python 3.9+
#   - pip install dvc[s3]
#   - MinIO running locally (docker-compose up -d minio)
#
# Usage:
#   chmod +x scripts/dvc_setup.sh
#   ./scripts/dvc_setup.sh
#
# Author: Trading Team
# Date: 2026-01-14

set -e

echo "=========================================="
echo "DVC Setup for USDCOP RL Trading System"
echo "=========================================="

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "DVC not found. Installing..."
    pip install "dvc[s3]"
fi

# Check DVC version
DVC_VERSION=$(dvc version | head -n1)
echo "Using: $DVC_VERSION"

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
fi

# Configure MinIO remote (local S3-compatible storage)
echo "Configuring MinIO remote..."
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id minioadmin
dvc remote modify --local minio secret_access_key minioadmin123

# Set as default remote
dvc remote default minio

echo ""
echo "Tracking critical files with DVC..."

# Track normalization stats (SSOT)
if [ -f "config/norm_stats.json" ]; then
    echo "  - config/norm_stats.json"
    dvc add config/norm_stats.json
fi

# Track feature config (SSOT)
if [ -f "config/feature_config.json" ]; then
    echo "  - config/feature_config.json"
    dvc add config/feature_config.json
fi

# Track trained models directory
if [ -d "models/ppo_primary" ]; then
    echo "  - models/ppo_primary/"
    dvc add models/ppo_primary
fi

if [ -d "models/ppo_production" ]; then
    echo "  - models/ppo_production/"
    dvc add models/ppo_production
fi

echo ""
echo "=========================================="
echo "DVC Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start MinIO: docker-compose up -d minio"
echo "  2. Create bucket: mc mb minio/dvc-storage"
echo "  3. Push data: dvc push"
echo "  4. Commit .dvc files: git add *.dvc .gitignore && git commit -m 'Add DVC tracking'"
echo ""
echo "To restore data on another machine:"
echo "  git clone <repo>"
echo "  dvc pull"
echo ""
