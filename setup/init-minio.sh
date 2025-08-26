#!/bin/bash

# Wait for MinIO to be ready
sleep 10

# Configure MinIO client
mc alias set minio http://minio:9000 minioadmin minioadmin

# Create all required buckets
mc mb minio/ds-usdcop-acquire -p
mc mb minio/ds-usdcop-standardize -p
mc mb minio/ds-usdcop-prepare -p
mc mb minio/ds-usdcop-feature -p
mc mb minio/ds-usdcop-mlready -p
mc mb minio/ds-usdcop-serving -p

# Set public read policy for serving bucket (optional)
mc anonymous set download minio/ds-usdcop-serving

echo "✅ All MinIO buckets created successfully"

# List buckets to confirm
mc ls minio/