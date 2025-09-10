#!/usr/bin/env python3
"""
Initialize MinIO buckets for USDCOP Trading System
"""

import json
import os
from datetime import datetime, timedelta
from minio import Minio
from minio.error import S3Error
import io

# Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"

def create_buckets():
    """Create all required buckets"""
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    buckets = [
        "00-raw-usdcop-marketdata",
        "01-l1-ds-usdcop-standardize",
        "02-l2-ds-usdcop-prepare",
        "03-l3-ds-usdcop-feature",
        "04-l4-ds-usdcop-rlready",
        "05-l5-ds-usdcop-serving",
        "99-common-trading-reports"
    ]
    
    for bucket_name in buckets:
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                print(f"Created bucket: {bucket_name}")
            else:
                print(f"Bucket already exists: {bucket_name}")
        except S3Error as e:
            print(f"Error creating bucket {bucket_name}: {e}")
    
    return client

def populate_sample_data(client):
    """Populate buckets with sample data"""
    
    # Generate sample L0 raw data
    sample_data = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "USD/COP",
        "bid": 4320.50,
        "ask": 4321.00,
        "mid": 4320.75,
        "volume": 1000000,
        "source": "mt5"
    }
    
    # L0 - Raw market data
    data_json = json.dumps(sample_data).encode('utf-8')
    client.put_object(
        "00-raw-usdcop-marketdata",
        f"data/{datetime.now().strftime('%Y%m%d')}/sample.json",
        io.BytesIO(data_json),
        len(data_json),
        content_type="application/json"
    )
    print("Populated L0 raw data")
    
    # L1 - Standardized data
    standardized_data = {
        **sample_data,
        "utc_timestamp": datetime.utcnow().isoformat(),
        "normalized_volume": 1.0
    }
    data_json = json.dumps(standardized_data).encode('utf-8')
    client.put_object(
        "01-l1-ds-usdcop-standardize",
        f"data/{datetime.now().strftime('%Y%m%d')}/standardized.json",
        io.BytesIO(data_json),
        len(data_json),
        content_type="application/json"
    )
    print("Populated L1 standardized data")
    
    # L2 - Prepared data
    prepared_data = {
        **standardized_data,
        "cleaned": True,
        "outliers_removed": 0
    }
    data_json = json.dumps(prepared_data).encode('utf-8')
    client.put_object(
        "02-l2-ds-usdcop-prepare",
        f"data/{datetime.now().strftime('%Y%m%d')}/prepared.json",
        io.BytesIO(data_json),
        len(data_json),
        content_type="application/json"
    )
    print("Populated L2 prepared data")
    
    # L3 - Feature data
    feature_data = {
        **prepared_data,
        "rsi": 45.5,
        "macd": 0.002,
        "bollinger_upper": 4330.0,
        "bollinger_lower": 4310.0,
        "correlation_matrix": [[1.0, 0.8], [0.8, 1.0]]
    }
    data_json = json.dumps(feature_data).encode('utf-8')
    client.put_object(
        "03-l3-ds-usdcop-feature",
        f"data/{datetime.now().strftime('%Y%m%d')}/features.json",
        io.BytesIO(data_json),
        len(data_json),
        content_type="application/json"
    )
    print("Populated L3 feature data")
    
    # L4 - RL Ready data
    rl_data = {
        "state": [4320.75, 45.5, 0.002],
        "action_space": [-1, 0, 1],
        "reward": 0.0
    }
    data_json = json.dumps(rl_data).encode('utf-8')
    client.put_object(
        "04-l4-ds-usdcop-rlready",
        f"data/{datetime.now().strftime('%Y%m%d')}/rlready.json",
        io.BytesIO(data_json),
        len(data_json),
        content_type="application/json"
    )
    print("Populated L4 RL-ready data")
    
    # L5 - Serving data
    serving_data = {
        "prediction": "BUY",
        "confidence": 0.75,
        "model_version": "v1.0.0",
        "timestamp": datetime.now().isoformat()
    }
    data_json = json.dumps(serving_data).encode('utf-8')
    client.put_object(
        "05-l5-ds-usdcop-serving",
        f"data/{datetime.now().strftime('%Y%m%d')}/predictions.json",
        io.BytesIO(data_json),
        len(data_json),
        content_type="application/json"
    )
    print("Populated L5 serving data")
    
    # L6 - Backtest reports
    report_data = {
        "strategy": "RL_PPO",
        "period": "2024-01-01 to 2024-12-31",
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.08,
        "win_rate": 0.55
    }
    data_json = json.dumps(report_data).encode('utf-8')
    client.put_object(
        "99-common-trading-reports",
        f"backtest/{datetime.now().strftime('%Y%m%d')}/report.json",
        io.BytesIO(data_json),
        len(data_json),
        content_type="application/json"
    )
    print("Populated L6 backtest reports")

def main():
    print("Initializing MinIO buckets for USDCOP Trading System...")
    
    try:
        client = create_buckets()
        populate_sample_data(client)
        print("\nAll buckets created and populated successfully!")
        print("MinIO is ready for the trading system.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())