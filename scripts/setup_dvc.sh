#!/bin/bash
# =============================================================================
# DVC Setup and Initialization Script
# =============================================================================
# Initializes DVC for the USDCOP trading project with MinIO remote storage.
# Updated as part of MLOps-2 implementation from remediation plan.
#
# Prerequisites:
#   - Python 3.11+ with pip
#   - MinIO running on localhost:9000 (or set MINIO_ENDPOINT)
#   - AWS credentials set for MinIO access
#
# Usage:
#   chmod +x scripts/setup_dvc.sh
#   ./scripts/setup_dvc.sh
#
# Author: Trading Team
# Date: 2026-01-16
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "DVC Setup for USDCOP Trading Project"
echo "=============================================="
echo ""

# Configuration (can be overridden via environment variables)
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
DVC_REMOTE_NAME="${DVC_REMOTE_NAME:-minio}"
DVC_BUCKET="${DVC_BUCKET:-dvc-storage}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from project root
if [ ! -f "dvc.yaml" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# =============================================================================
# Step 1: Install DVC with S3 support
# =============================================================================
log_info "Checking DVC installation..."

if ! command -v dvc &> /dev/null; then
    log_info "DVC not found. Installing with S3 support..."
    pip install --quiet --upgrade "dvc[s3]>=3.42.0"
else
    log_info "DVC is already installed: $(dvc --version)"
fi

# =============================================================================
# Step 2: Initialize DVC (if not already initialized)
# =============================================================================
if [ -d ".dvc" ]; then
    log_info "DVC already initialized, skipping..."
else
    log_info "Initializing DVC..."
    dvc init
    log_info "DVC initialized successfully"
fi

# =============================================================================
# Step 3: Configure MinIO remote
# =============================================================================
log_info "Configuring MinIO remote storage..."

# Remove existing remotes and configure MinIO
if dvc remote list | grep -q "$DVC_REMOTE_NAME"; then
    log_info "Updating existing remote '$DVC_REMOTE_NAME'..."
    dvc remote remove $DVC_REMOTE_NAME
fi

# Also remove legacy 'myremote' if it exists
if dvc remote list | grep -q "myremote"; then
    log_warn "Removing legacy 'myremote' remote..."
    dvc remote remove myremote
fi

# Add MinIO as S3-compatible remote
dvc remote add -d $DVC_REMOTE_NAME s3://$DVC_BUCKET

# Configure MinIO endpoint
dvc remote modify $DVC_REMOTE_NAME endpointurl $MINIO_ENDPOINT

# Additional S3 settings for MinIO compatibility
dvc remote modify $DVC_REMOTE_NAME use_ssl false

log_info "Remote '$DVC_REMOTE_NAME' configured with:"
log_info "  - Bucket: s3://$DVC_BUCKET"
log_info "  - Endpoint: $MINIO_ENDPOINT"

# =============================================================================
# Step 4: Create required directories
# =============================================================================
log_info "Creating required directories..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/onnx
mkdir -p reports/backtest

# Create .gitkeep files
touch data/raw/.gitkeep 2>/dev/null || true
touch data/processed/.gitkeep 2>/dev/null || true
touch models/onnx/.gitkeep 2>/dev/null || true
touch reports/.gitkeep 2>/dev/null || true

log_info "Directories created"

# =============================================================================
# Step 5: Create .dvcignore
# =============================================================================
if [ ! -f ".dvcignore" ]; then
    log_info "Creating .dvcignore..."
    cat > .dvcignore << 'EOF'
# =============================================================================
# DVC Ignore Patterns
# =============================================================================
# Similar to .gitignore but for DVC file tracking.

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Temporary files
*.tmp
*.temp
*.log
*.bak

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Test artifacts
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# Build artifacts
build/
dist/
*.egg-info/

# Notebooks
.ipynb_checkpoints/

# Node modules (dashboard)
node_modules/
.next/

# Git
.git/

# DVC internal
.dvc/cache/
.dvc/tmp/
EOF
    log_info ".dvcignore created"
else
    log_info ".dvcignore already exists"
fi

# =============================================================================
# Step 6: Verify configuration
# =============================================================================
log_info "Verifying DVC configuration..."

echo ""
echo "Current DVC remotes:"
echo "-------------------------------------------"
dvc remote list

echo ""
echo "DVC configuration:"
echo "-------------------------------------------"
dvc config -l | grep -E "^remote\." || echo "  (using defaults)"

# =============================================================================
# Step 7: Add DVC files to git
# =============================================================================
log_info "Adding DVC files to git staging..."
git add .dvc/config .dvcignore 2>/dev/null || true
git add .dvc/.gitignore 2>/dev/null || true

# =============================================================================
# Step 8: Create local config template
# =============================================================================
if [ ! -f ".dvc/config.local" ] && [ ! -f ".dvc/config.local.example" ]; then
    log_info "Creating local DVC config template..."
    cat > .dvc/config.local.example << 'EOF'
# =============================================================================
# Local DVC Configuration (not tracked by git)
# =============================================================================
# Copy this file to .dvc/config.local and modify as needed.
# Local settings override the main config.

[core]
    # Auto-stage DVC files for git after dvc add/run
    autostage = true

# Uncomment to use a shared cache directory
# [cache]
#     dir = /path/to/shared/cache

# Uncomment for faster operations (skip hash verification)
# [remote "minio"]
#     verify = false

# Uncomment to use environment variables for credentials
# [remote "minio"]
#     access_key_id = ${AWS_ACCESS_KEY_ID}
#     secret_access_key = ${AWS_SECRET_ACCESS_KEY}
EOF
    log_info "Created .dvc/config.local.example"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "DVC Setup Complete!"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Remote: $DVC_REMOTE_NAME"
echo "  Bucket: s3://$DVC_BUCKET"
echo "  Endpoint: $MINIO_ENDPOINT"
echo ""
echo "Next steps:"
echo ""
echo "1. Set MinIO credentials (if not already set):"
echo "   export AWS_ACCESS_KEY_ID=your_minio_access_key"
echo "   export AWS_SECRET_ACCESS_KEY=your_minio_secret_key"
echo ""
echo "2. Test the remote connection:"
echo "   dvc push --dry"
echo ""
echo "3. Add data files to DVC tracking:"
echo "   dvc add data/raw/your_data_file.parquet"
echo ""
echo "4. Commit the .dvc files to git:"
echo "   git add data/raw/your_data_file.parquet.dvc .gitignore"
echo "   git commit -m 'Add data file to DVC'"
echo ""
echo "5. Push data to remote storage:"
echo "   dvc push"
echo ""
echo "6. Run the ML pipeline:"
echo "   dvc repro"
echo ""
echo "Useful commands:"
echo "  dvc status       - Check status of tracked files"
echo "  dvc diff HEAD~1  - Show changes between versions"
echo "  dvc dag          - Visualize pipeline DAG"
echo "  dvc metrics show - Show pipeline metrics"
echo "  dvc plots show   - Generate pipeline plots"
echo ""
echo "For more information, see: https://dvc.org/doc"
echo ""

log_info "Setup complete!"
