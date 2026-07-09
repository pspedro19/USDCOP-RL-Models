#!/bin/bash
# =============================================================================
# rollback_dataset.sh - DVC Dataset Rollback Script
# =============================================================================
# Contract: DVC-32
# Purpose: Rollback to a previous dataset version using DVC and git
#
# Usage:
#   ./scripts/rollback_dataset.sh [options]
#
# Options:
#   -t, --tag TAG       Git tag to rollback to (e.g., v1.0.0)
#   -c, --commit SHA    Git commit SHA to rollback to
#   -l, --list          List available tags/versions
#   -r, --remote REMOTE DVC remote to pull from (default: minio)
#   -f, --force         Force rollback even with uncommitted changes
#   -d, --dry-run       Show what would be done without executing
#   -h, --help          Show this help message
#
# Example:
#   ./scripts/rollback_dataset.sh -t v1.0.0
#   ./scripts/rollback_dataset.sh -c abc123
#   ./scripts/rollback_dataset.sh -l
#
# Prerequisites:
#   - DVC installed and configured
#   - Git repository with history
#   - DVC remote accessible
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
COMMIT=""
LIST_VERSIONS=false

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
    head -35 "$0" | grep -E "^#" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -c|--commit)
            COMMIT="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        -l|--list)
            LIST_VERSIONS=true
            shift
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

# Change to project root
cd "$PROJECT_ROOT"

# List versions mode
if [[ "$LIST_VERSIONS" == true ]]; then
    log_info "Available dataset versions:"
    log_info ""
    log_info "Tags:"
    git tag -l --sort=-version:refname | head -20 | while read -r tag; do
        TAG_DATE=$(git log -1 --format="%ci" "$tag" 2>/dev/null || echo "unknown")
        echo "  $tag ($TAG_DATE)"
    done
    log_info ""
    log_info "Recent commits with DVC changes:"
    git log --oneline --all -- "*.dvc" "dvc.lock" | head -15 | while read -r line; do
        echo "  $line"
    done
    exit 0
fi

# Validate input
if [[ -z "$TAG" && -z "$COMMIT" ]]; then
    log_error "Must specify either --tag or --commit"
    log_info "Use --list to see available versions"
    exit 1
fi

# Determine target ref
TARGET_REF=""
if [[ -n "$TAG" ]]; then
    if ! git rev-parse "$TAG" >/dev/null 2>&1; then
        log_error "Tag '$TAG' not found"
        log_info "Available tags:"
        git tag -l | head -10
        exit 1
    fi
    TARGET_REF="$TAG"
    log_info "Rolling back to tag: $TAG"
elif [[ -n "$COMMIT" ]]; then
    if ! git rev-parse "$COMMIT" >/dev/null 2>&1; then
        log_error "Commit '$COMMIT' not found"
        exit 1
    fi
    TARGET_REF="$COMMIT"
    log_info "Rolling back to commit: $COMMIT"
fi

# Get current state for backup reference
CURRENT_COMMIT=$(git rev-parse HEAD)
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
log_info "Current state: $CURRENT_BRANCH @ ${CURRENT_COMMIT:0:8}"

# Step 1: Check for uncommitted changes
log_info "Checking for uncommitted changes..."
if [[ -n "$(git status --porcelain)" ]]; then
    log_warning "You have uncommitted changes:"
    git status --short
    if [[ "$FORCE" != true ]]; then
        log_error "Stash or commit changes before rollback. Use -f to force."
        exit 1
    fi
    log_warning "Proceeding anyway due to --force flag"
fi

# Step 2: Create backup branch
BACKUP_BRANCH="backup-before-rollback-$(date +%Y%m%d-%H%M%S)"
log_info "Creating backup branch: $BACKUP_BRANCH"
if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would create branch: $BACKUP_BRANCH"
else
    git branch "$BACKUP_BRANCH" 2>/dev/null || log_warning "Backup branch already exists"
    log_success "Backup created: $BACKUP_BRANCH"
fi

# Step 3: Checkout DVC files from target version
log_info "Checking out DVC files from $TARGET_REF..."
DVC_FILES="dvc.lock dvc.yaml"
for file in $DVC_FILES; do
    if git show "$TARGET_REF:$file" >/dev/null 2>&1; then
        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY-RUN] Would checkout: $file"
        else
            git checkout "$TARGET_REF" -- "$file"
            log_success "Checked out: $file"
        fi
    else
        log_warning "File not found in target: $file"
    fi
done

# Checkout .dvc files
log_info "Checking out .dvc tracked files..."
git ls-tree -r --name-only "$TARGET_REF" | grep "\.dvc$" | while read -r dvc_file; do
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would checkout: $dvc_file"
    else
        git checkout "$TARGET_REF" -- "$dvc_file" 2>/dev/null || log_warning "Could not checkout: $dvc_file"
    fi
done

# Step 4: Pull data from DVC remote
log_info "Pulling data from DVC remote: $REMOTE"
if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would run: dvc checkout"
    log_info "[DRY-RUN] Would run: dvc pull -r $REMOTE"
else
    # First checkout to update .dvc cache references
    dvc checkout

    # Then pull actual data from remote
    dvc pull -r "$REMOTE"
    log_success "Data pulled from remote"
fi

# Step 5: Verify data integrity
log_info "Verifying data integrity..."
if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would verify data with: dvc status"
else
    DVC_STATUS=$(dvc status 2>/dev/null || echo "")
    if [[ -z "$DVC_STATUS" || "$DVC_STATUS" == *"Data and calculation are the same"* ]]; then
        log_success "Data integrity verified"
    else
        log_warning "DVC status: $DVC_STATUS"
    fi
fi

# Step 6: Compute hash for verification
log_info "Computing dataset hash..."
DATASET_HASH=""
if [[ -f "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv" ]]; then
    DATASET_HASH=$(sha256sum "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv" | cut -d' ' -f1)
    log_info "Current dataset hash: ${DATASET_HASH:0:16}..."
fi

# Step 7: Create rollback commit
ROLLBACK_MESSAGE="Rollback dataset to $TARGET_REF

Previous state: ${CURRENT_COMMIT:0:8}
Backup branch: $BACKUP_BRANCH
Dataset hash: ${DATASET_HASH:0:16}
DVC remote: $REMOTE
Rollback timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

Co-Authored-By: DVC Pipeline <noreply@dvc.org>"

if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] Would commit rollback with message:"
    echo "$ROLLBACK_MESSAGE"
else
    git add *.dvc dvc.lock 2>/dev/null || true
    git commit -m "$ROLLBACK_MESSAGE" || log_warning "Nothing to commit (files unchanged)"
fi

# Summary
log_info "=========================================="
log_info "DATASET ROLLBACK SUMMARY"
log_info "=========================================="
log_info "Target Version: $TARGET_REF"
log_info "Backup Branch:  $BACKUP_BRANCH"
log_info "Dataset Hash:   ${DATASET_HASH:0:16}..."
log_info "DVC Remote:     $REMOTE"
log_info "Dry Run:        $DRY_RUN"
log_info "Timestamp:      $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
log_success "=========================================="
log_success "Dataset rollback completed!"
log_info ""
log_info "To undo this rollback, run:"
log_info "  git checkout $BACKUP_BRANCH -- dvc.lock *.dvc && dvc checkout"
