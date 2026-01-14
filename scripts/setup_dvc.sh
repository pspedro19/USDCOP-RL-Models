#!/bin/bash
# =============================================================================
# DVC Setup Script (Phase 11)
# =============================================================================
# Setup DVC for data versioning in the USDCOP trading system.
#
# Usage:
#   chmod +x scripts/setup_dvc.sh
#   ./scripts/setup_dvc.sh
#
# Author: Trading Team
# Date: 2025-01-14
# =============================================================================

set -e

echo "=== Setting up DVC for Data Versioning ==="
echo ""

# =============================================================================
# 1. Check if DVC is installed
# =============================================================================
if ! command -v dvc &> /dev/null; then
    echo "DVC not found. Installing..."
    pip install "dvc[s3]"  # Includes S3 support
else
    echo "DVC is already installed: $(dvc --version)"
fi

# =============================================================================
# 2. Initialize DVC (if not already initialized)
# =============================================================================
if [ ! -d ".dvc" ]; then
    echo ""
    echo "Initializing DVC..."
    dvc init
    echo "DVC initialized successfully"
else
    echo "DVC already initialized (.dvc directory exists)"
fi

# =============================================================================
# 3. Configure remote storage
# =============================================================================
echo ""
echo "Configuring remote storage..."

# Check if remote already exists
if dvc remote list | grep -q "myremote"; then
    echo "Remote 'myremote' already configured"
else
    # Default to local remote for initial setup
    # Users can reconfigure to S3/GCS later
    mkdir -p .dvc/remote-storage
    dvc remote add -d myremote .dvc/remote-storage
    echo "Added local remote at .dvc/remote-storage"
    echo ""
    echo "To use S3 instead, run:"
    echo "  dvc remote modify myremote url s3://your-bucket/dvc-store"
    echo "  dvc remote modify myremote region us-east-1"
fi

# =============================================================================
# 4. Track key data files
# =============================================================================
echo ""
echo "Tracking data files..."

# Track normalization stats (critical for model parity)
if [ -f "config/norm_stats.json" ]; then
    if [ ! -f "config/norm_stats.json.dvc" ]; then
        dvc add config/norm_stats.json
        echo "Added config/norm_stats.json to DVC tracking"
    else
        echo "config/norm_stats.json already tracked"
    fi
fi

# Track training datasets if they exist
if [ -d "data/training" ]; then
    for file in data/training/*.parquet; do
        if [ -f "$file" ] && [ ! -f "${file}.dvc" ]; then
            dvc add "$file"
            echo "Added $file to DVC tracking"
        fi
    done
fi

# =============================================================================
# 5. Create .dvcignore
# =============================================================================
if [ ! -f ".dvcignore" ]; then
    cat > .dvcignore << 'EOF'
# DVC ignore patterns
# Similar to .gitignore but for DVC

# Python cache
__pycache__/
*.pyc
*.pyo

# Temporary files
*.tmp
*.temp
*.log

# IDE files
.idea/
.vscode/
*.swp

# Test artifacts
.pytest_cache/
.coverage
htmlcov/

# Build artifacts
build/
dist/
*.egg-info/
EOF
    echo "Created .dvcignore"
fi

# =============================================================================
# 6. Add DVC files to git
# =============================================================================
echo ""
echo "Adding DVC files to git..."
git add .dvc/config .dvcignore 2>/dev/null || true
git add *.dvc 2>/dev/null || true
git add **/*.dvc 2>/dev/null || true

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=== DVC Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Commit DVC configuration: git commit -m 'Initialize DVC for data versioning'"
echo "2. Push data to remote: dvc push"
echo "3. To pull data on another machine: dvc pull"
echo ""
echo "Useful commands:"
echo "  dvc status       - Check status of tracked files"
echo "  dvc diff HEAD~1  - Show changes between versions"
echo "  dvc push         - Push data to remote storage"
echo "  dvc pull         - Pull data from remote storage"
echo "  dvc repro        - Reproduce pipeline stages"
echo ""
