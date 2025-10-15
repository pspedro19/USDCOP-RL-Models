#!/bin/bash

# MinIO Bucket Initialization Script
# This script creates all required buckets for the USDCOP trading system

set -e  # Exit on any error

echo "=== MinIO Bucket Initialization Started ==="
echo "Timestamp: $(date)"

# Configuration
MINIO_ENDPOINT="http://minio:9000"
MAX_RETRIES=30
RETRY_DELAY=2

# Function to wait for MinIO to be ready
wait_for_minio() {
    echo "Waiting for MinIO server to be ready..."
    local retries=0

    while [ $retries -lt $MAX_RETRIES ]; do
        if mc ping $MINIO_ENDPOINT --exit-code > /dev/null 2>&1; then
            echo "MinIO server is ready!"
            return 0
        fi

        retries=$((retries + 1))
        echo "Attempt $retries/$MAX_RETRIES: MinIO not ready yet, waiting ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done

    echo "ERROR: MinIO server is not ready after $MAX_RETRIES attempts"
    return 1
}

# Function to create bucket with error handling
create_bucket() {
    local bucket_name=$1
    echo "Creating bucket: $bucket_name"

    if mc mb --ignore-existing minio/$bucket_name; then
        echo "✓ Bucket '$bucket_name' created successfully"
        return 0
    else
        echo "✗ Failed to create bucket '$bucket_name'"
        return 1
    fi
}

# Main initialization process
main() {
    echo "Setting up MinIO alias..."
    if ! mc alias set minio $MINIO_ENDPOINT ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}; then
        echo "ERROR: Failed to set MinIO alias"
        exit 1
    fi
    echo "✓ MinIO alias configured successfully"

    # Wait for MinIO to be fully ready
    if ! wait_for_minio; then
        exit 1
    fi

    # List of buckets to create
    buckets=(
        "00-raw-usdcop-marketdata"
        "01-l1-ds-usdcop-standardize"
        "02-l2-ds-usdcop-prepare"
        "03-l3-ds-usdcop-feature"
        "04-l4-ds-usdcop-rlready"
        "05-l5-ds-usdcop-serving"
        "usdcop-l4-rlready"
        "usdcop-l5-serving"
        "usdcop-l6-backtest"
        "99-common-trading-models"
        "99-common-trading-reports"
        "99-common-trading-backups"
    )

    echo "Creating ${#buckets[@]} buckets..."
    failed_buckets=()

    for bucket in "${buckets[@]}"; do
        if ! create_bucket "$bucket"; then
            failed_buckets+=("$bucket")
        fi
    done

    # Report results
    echo ""
    echo "=== Initialization Summary ==="
    if [ ${#failed_buckets[@]} -eq 0 ]; then
        echo "✓ All ${#buckets[@]} buckets created successfully!"
        echo "Verifying buckets..."
        mc ls minio/ || echo "Warning: Could not list buckets for verification"
    else
        echo "✗ Failed to create ${#failed_buckets[@]} buckets:"
        printf '  - %s\n' "${failed_buckets[@]}"
        exit 1
    fi

    echo "=== MinIO Bucket Initialization Completed ==="
    echo "Timestamp: $(date)"
}

# Run main function
main "$@"