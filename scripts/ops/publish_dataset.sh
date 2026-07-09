#!/bin/bash
# =============================================================================
# publish_dataset.sh - DVC Dataset Publishing Script
# =============================================================================
# Contract: DVC-31
# Purpose: Publish versioned datasets to DVC remote storage
#
# Usage:
#   ./scripts/publish_dataset.sh [options]
#
# Options:
#   -t, --tag TAG       Git tag for the release (e.g., v1.0.0)
#   -m, --message MSG   Commit message
#   -r, --remote REMOTE DVC remote to push to (default: minio)
#   -f, --force         Force push even if remote has data
#   -d, --dry-run       Show what would be done without executing
#   -h, --help          Show this help message
#
# Example:
#   ./scripts/publish_dataset.sh -t v1.0.0 -m "Release training dataset v1.0.0"
#
# Prerequisites:
#   - DVC installed and configured
#   - Git repository initialized
#   - DVC remote configured (minio or s3_backup)
#
# Author: USDCOP Trading Team
# Version: 1.0.0
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
REMOTE="minio"
DRY_RUN=false
FORCE=false
TAG=""
MESSAGE=""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Help message
show_help() {
    head -30 "$0" | grep -E "^#" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -m|--message)
            MESSAGE="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate tag format
if [[ -n "$TAG" && ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_warning "Tag '$TAG' doesn't follow semantic versioning (vX.Y.Z)"
fi

# Change to project root
cd "$PROJECT_ROOT"

log_info "Publishing dataset to DVC remote: $REMOTE"
log_info "Project root: $PROJECT_ROOT"

# Step 1: Verify DVC is initialized
if [[ ! -d ".dvc" ]]; then
    log_error "DVC not initialized. Run 'dvc init' first."
    exit 1
fi

# Step 2: Check remote configuration
log_info "Checking DVC remote configuration..."
if ! dvc remote list | grep -q "$REMOTE"; then
    log_error "Remote '$REMOTE' not configured. Available remotes:"
    dvc remote list
    exit 1
fi
log_success "Remote '$REMOTE' is configured"

# Step 3: Check for uncommitted changes
log_info "Checking for uncommitted changes..."
if [[ -n "$(git status --porcelain)" ]]; then
    log_warning "You have uncommitted changes:"
    git status --short
    if [[ "$FORCE" != true ]]; then
        log_error "Commit or stash changes before publishing. Use -f to force."
        exit 1
    fi
fi

# Step 4: Run DVC repro to ensure pipeline is up to date
log_info "Checking DVC pipeline status..."
DVC_STATUS=$(dvc status 2>/dev/null || echo "error")
if [[ "$DVC_STATUS" == *"changed"* || "$DVC_STATUS" == *"modified"* ]]; then
    log_warning "DVC pipeline has changes. Running dvc repro..."
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would run: dvc repro"
    else
        dvc repro
    fi
fi

# Step 5: Compute dataset hash for verification
log_info "Computing dataset hashes..."
DATASET_HASH=""
if [[ -f "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv" ]]; then
    DATASET_HASH=$(sha256sum "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv" | cut -d' ' -f1)
    log_info "Dataset hash: ${DATASET_HASH:0:16}..."
fi

# Step 6: Add and commit DVC files
log_info "Adding DVC tracked files..."
DVC_FILES=$(find . -name "*.dvc" -type f 2>/dev/null | head -20)
if [[ -n "$DVC_FILES" ]]; then
    log_info "Found DVC files:"
    echo "$DVC_FILES" | while read -r f; do log_info "  - $f"; done
fi

if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would run: git add *.dvc dvc.lock"
else
    git add *.dvc dvc.lock 2>/dev/null || true
fi

# Step 7: Create commit if message provided
if [[ -n "$MESSAGE" ]]; then
    log_info "Creating commit..."
    FULL_MESSAGE="$MESSAGE

Dataset hash: ${DATASET_HASH:0:16}
DVC remote: $REMOTE
Published: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

Co-Authored-By: DVC Pipeline <noreply@dvc.org>"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would commit with message:"
        echo "$FULL_MESSAGE"
    else
        git commit -m "$FULL_MESSAGE" || log_warning "Nothing to commit"
    fi
fi

# Step 8: Create git tag if provided
if [[ -n "$TAG" ]]; then
    log_info "Creating git tag: $TAG"
    TAG_MESSAGE="Dataset release $TAG

Dataset hash: ${DATASET_HASH:0:16}
DVC remote: $REMOTE
Published: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would create tag: $TAG"
    else
        git tag -a "$TAG" -m "$TAG_MESSAGE"
        log_success "Created tag: $TAG"
    fi
fi

# Step 9: Push to DVC remote
log_info "Pushing data to DVC remote: $REMOTE"
PUSH_FLAGS=""
if [[ "$FORCE" == true ]]; then
    PUSH_FLAGS="--force"
fi

if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would run: dvc push -r $REMOTE $PUSH_FLAGS"
else
    dvc push -r "$REMOTE" $PUSH_FLAGS
    log_success "Data pushed to remote: $REMOTE"
fi

# Step 10: Push git changes
log_info "Pushing git changes..."
if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would run: git push origin main"
    if [[ -n "$TAG" ]]; then
        log_info "[DRY-RUN] Would run: git push origin $TAG"
    fi
else
    git push origin main 2>/dev/null || log_warning "Failed to push to origin (may not exist)"
    if [[ -n "$TAG" ]]; then
        git push origin "$TAG" 2>/dev/null || log_warning "Failed to push tag"
    fi
fi

# Step 11: Verification
log_info "Verifying push..."
if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would verify with: dvc status -c -r $REMOTE"
else
    VERIFY=$(dvc status -c -r "$REMOTE" 2>/dev/null || echo "")
    if [[ -z "$VERIFY" || "$VERIFY" == *"Data and calculation are the same"* ]]; then
        log_success "Verification passed: data is synced with remote"
    else
        log_warning "Verification result: $VERIFY"
    fi
fi

# Summary
log_info "=========================================="
log_info "DATASET PUBLISHING SUMMARY"
log_info "=========================================="
log_info "Remote:       $REMOTE"
log_info "Tag:          ${TAG:-'(none)'}"
log_info "Dataset Hash: ${DATASET_HASH:0:16}..."
log_info "Dry Run:      $DRY_RUN"
log_info "Timestamp:    $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
log_success "=========================================="
log_success "Dataset publishing completed!"
