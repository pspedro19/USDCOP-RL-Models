"""
Download L3 features from MinIO to local for L4 processing
"""

from minio import Minio
from io import BytesIO
import pandas as pd

# MinIO configuration
MINIO_CLIENT = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

def download_l3_features():
    """Download L3 features from MinIO"""
    
    print("Downloading L3 features from MinIO...")
    
    bucket_name = 'ds-usdcop-feature'
    
    # Try to find the largest features file
    paths = [
        'usdcop_m5__04_l3_feature/latest/features.csv',
        'temp/l3_feature/L3_20250822_045813/all_features.parquet',
        'usdcop_m5__04_l3_feature/market=usdcop/timeframe=m5/date=2025-08-22/run_id=L3_20250822_045813/features.csv'
    ]
    
    for path in paths:
        try:
            print(f"  Trying: {path}")
            response = MINIO_CLIENT.get_object(bucket_name, path)
            data = response.read()
            response.close()
            
            if '.parquet' in path:
                df = pd.read_parquet(BytesIO(data))
            else:
                df = pd.read_csv(BytesIO(data))
            
            if len(df) > 50000:  # We expect ~53,640 rows
                print(f"  Found: {len(df):,} rows")
                
                # Save locally
                output_path = 'data/processed/gold/USDCOP_gold_features.csv'
                df.to_csv(output_path, index=False)
                print(f"  Saved to: {output_path}")
                return True
                
        except Exception as e:
            continue
    
    print("  Could not find L3 features in MinIO")
    return False

if __name__ == "__main__":
    download_l3_features()