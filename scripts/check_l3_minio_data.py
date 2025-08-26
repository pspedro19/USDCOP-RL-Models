"""
Check L3 data availability in MinIO
"""

from minio import Minio
from datetime import datetime

# MinIO configuration
MINIO_CLIENT = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

def check_l3_data():
    """Check what L3 data is available in MinIO"""
    
    print("="*80)
    print("Checking L3 Data in MinIO")
    print("="*80)
    
    # List all buckets
    buckets = MINIO_CLIENT.list_buckets()
    print("\nAvailable buckets:")
    for bucket in buckets:
        print(f"  - {bucket.name}")
    
    # Check L3 buckets
    l3_buckets = ['ds-usdcop-features', 'usdcop-features', 'ds-l3-features']
    
    for bucket_name in l3_buckets:
        try:
            # Check if bucket exists
            if MINIO_CLIENT.bucket_exists(bucket_name):
                print(f"\n[OK] Bucket '{bucket_name}' exists")
                
                # List objects in bucket
                objects = list(MINIO_CLIENT.list_objects(bucket_name, recursive=True))
                
                if objects:
                    print(f"  Found {len(objects)} objects:")
                    
                    # Show first 10 objects
                    for i, obj in enumerate(objects[:10]):
                        print(f"    {i+1}. {obj.object_name} ({obj.size:,} bytes)")
                    
                    if len(objects) > 10:
                        print(f"    ... and {len(objects) - 10} more objects")
                    
                    # Check for specific L3 files
                    l3_files = ['features.parquet', 'feature_spec.json', 'leakage_gate.json']
                    for file_name in l3_files:
                        found = any(file_name in obj.object_name for obj in objects)
                        if found:
                            print(f"  [Found] {file_name}")
                        else:
                            print(f"  [Missing] {file_name}")
                else:
                    print(f"  Bucket is empty")
        except Exception as e:
            print(f"\n[INFO] Bucket '{bucket_name}' not found or error: {e}")
    
    # Check for L2 prepare bucket (which feeds into L3)
    l2_bucket = 'ds-usdcop-prepared'
    if MINIO_CLIENT.bucket_exists(l2_bucket):
        print(f"\n[OK] L2 bucket '{l2_bucket}' exists")
        objects = list(MINIO_CLIENT.list_objects(l2_bucket, recursive=True))
        if objects:
            print(f"  Found {len(objects)} L2 objects (potential L3 inputs)")
            
            # Show latest dates
            dates = set()
            for obj in objects:
                if 'date=' in obj.object_name:
                    date_part = obj.object_name.split('date=')[1].split('/')[0]
                    dates.add(date_part)
            
            if dates:
                sorted_dates = sorted(dates)
                print(f"  Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
                print(f"  Total days with data: {len(dates)}")

if __name__ == "__main__":
    check_l3_data()