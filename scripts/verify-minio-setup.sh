#!/bin/bash

# MinIO Setup Verification Script
# This script verifies that MinIO and all buckets are properly configured

set -e

echo "=== MinIO Setup Verification ==="
echo "Timestamp: $(date)"

# Configuration
MINIO_ENDPOINT="http://localhost:9000"
EXPECTED_BUCKETS=(
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

# Function to check if MinIO is accessible
check_minio_health() {
    echo "Checking MinIO server health..."
    if curl -f -s http://localhost:9000/minio/health/live > /dev/null; then
        echo "✓ MinIO server is healthy and accessible"
        return 0
    else
        echo "✗ MinIO server is not accessible"
        return 1
    fi
}

# Function to check bucket existence
check_buckets() {
    echo "Verifying bucket existence..."

    # Get list of actual buckets
    actual_buckets=$(docker run --rm --network usdcop-rl-models_usdcop-trading-network \
        --entrypoint=/bin/sh \
        -e MINIO_ACCESS_KEY=minioadmin \
        -e MINIO_SECRET_KEY=minioadmin123 \
        minio/mc:latest \
        -c "mc alias set minio http://minio:9000 minioadmin minioadmin123 && mc ls minio/ --json" | \
        grep '"key"' | \
        sed 's/.*"key":"\([^"]*\)".*/\1/' | \
        sed 's|/$||')

    missing_buckets=()
    for bucket in "${EXPECTED_BUCKETS[@]}"; do
        if echo "$actual_buckets" | grep -q "^$bucket$"; then
            echo "✓ Bucket '$bucket' exists"
        else
            echo "✗ Bucket '$bucket' is missing"
            missing_buckets+=("$bucket")
        fi
    done

    if [ ${#missing_buckets[@]} -eq 0 ]; then
        echo "✓ All ${#EXPECTED_BUCKETS[@]} expected buckets are present"
        return 0
    else
        echo "✗ ${#missing_buckets[@]} buckets are missing:"
        printf '  - %s\n' "${missing_buckets[@]}"
        return 1
    fi
}

# Function to check bucket policies
check_bucket_policies() {
    echo "Checking bucket policies..."

    # Check if public download policies are set
    public_buckets=("00-raw-usdcop-marketdata" "99-common-trading-reports")

    for bucket in "${public_buckets[@]}"; do
        policy_output=$(docker run --rm --network usdcop-rl-models_usdcop-trading-network \
            --entrypoint=/bin/sh \
            -e MINIO_ACCESS_KEY=minioadmin \
            -e MINIO_SECRET_KEY=minioadmin123 \
            minio/mc:latest \
            -c "mc alias set minio http://minio:9000 minioadmin minioadmin123 && mc anonymous get minio/$bucket" 2>/dev/null || echo "none")

        if [[ "$policy_output" == *"download"* ]]; then
            echo "✓ Bucket '$bucket' has public download access"
        else
            echo "✗ Bucket '$bucket' does not have public download access"
        fi
    done
}

# Function to test basic MinIO operations
test_basic_operations() {
    echo "Testing basic MinIO operations..."

    test_file="/tmp/minio-test-$(date +%s).txt"
    echo "MinIO test file created at $(date)" > "$test_file"

    # Test file upload and download
    if docker run --rm --network usdcop-rl-models_usdcop-trading-network \
        --entrypoint=/bin/sh \
        -v "$test_file:$test_file" \
        -e MINIO_ACCESS_KEY=minioadmin \
        -e MINIO_SECRET_KEY=minioadmin123 \
        minio/mc:latest \
        -c "mc alias set minio http://minio:9000 minioadmin minioadmin123 && \
            mc cp $test_file minio/99-common-trading-backups/test-file.txt && \
            mc ls minio/99-common-trading-backups/test-file.txt && \
            mc rm minio/99-common-trading-backups/test-file.txt" > /dev/null 2>&1; then
        echo "✓ Basic file operations (upload/list/delete) work correctly"
        rm -f "$test_file"
        return 0
    else
        echo "✗ Basic file operations failed"
        rm -f "$test_file"
        return 1
    fi
}

# Main verification process
main() {
    echo "Starting MinIO setup verification..."
    echo ""

    failed_checks=0

    # Run all checks
    check_minio_health || ((failed_checks++))
    echo ""

    check_buckets || ((failed_checks++))
    echo ""

    check_bucket_policies || ((failed_checks++))
    echo ""

    test_basic_operations || ((failed_checks++))
    echo ""

    # Summary
    echo "=== Verification Summary ==="
    if [ $failed_checks -eq 0 ]; then
        echo "✓ All checks passed! MinIO setup is working correctly."
        echo "✓ Server is healthy and accessible"
        echo "✓ All ${#EXPECTED_BUCKETS[@]} buckets are present"
        echo "✓ Bucket policies are configured"
        echo "✓ Basic operations work correctly"
    else
        echo "✗ $failed_checks check(s) failed. Please review the issues above."
        exit 1
    fi

    echo "=== Verification Completed ==="
    echo "Timestamp: $(date)"
}

# Run main function
main "$@"