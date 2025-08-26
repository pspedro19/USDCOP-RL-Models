"""
Verify L4 files saved to MinIO
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

def verify_l4_files():
    """Verify L4 files in MinIO"""
    
    print("="*80)
    print("Verifying L4 Files in MinIO")
    print("="*80)
    
    bucket_name = 'ds-usdcop-rlready'
    run_id = 'L4_FULL_20250822_115113'
    date = '2025-08-22'
    
    base_path = f"usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5/date={date}/run_id={run_id}"
    
    # Expected files
    expected_files = [
        'replay_dataset.csv',
        'episodes_index.csv',
        'env_spec.json',
        'reward_spec.json',
        'cost_model.json',
        'action_spec.json',
        'split_spec.json',
        'checks_report.json',
        'metadata.json',
        '_control/READY'
    ]
    
    print(f"\nChecking run: {run_id}")
    print(f"Path: {base_path}")
    print("-"*80)
    
    # List objects in the path
    objects = list(MINIO_CLIENT.list_objects(bucket_name, prefix=base_path, recursive=True))
    
    if not objects:
        print("[ERROR] No objects found in the specified path")
        return False
    
    print(f"Found {len(objects)} objects:")
    
    found_files = {}
    for obj in objects:
        file_name = obj.object_name.replace(base_path + '/', '')
        found_files[file_name] = obj.size
        print(f"  [OK] {file_name} ({obj.size:,} bytes)")
    
    print("\nValidation:")
    all_found = True
    
    for expected in expected_files:
        if expected in found_files:
            print(f"  [OK] {expected}")
        else:
            print(f"  [MISSING] {expected}")
            all_found = False
    
    if all_found:
        print("\n[SUCCESS] ALL L4 FILES SUCCESSFULLY SAVED TO MINIO!")
    else:
        print("\n[WARNING] Some files are missing")
    
    # Show a sample of the metadata
    try:
        response = MINIO_CLIENT.get_object(bucket_name, f"{base_path}/metadata.json")
        metadata = response.read().decode('utf-8')
        response.close()
        
        print("\nMetadata Preview:")
        print("-"*40)
        import json
        meta = json.loads(metadata)
        print(f"Pipeline: {meta.get('pipeline')}")
        print(f"Version: {meta.get('version')}")
        print(f"Run ID: {meta.get('run_id')}")
        print(f"Date Range: {meta.get('temporal_range', {}).get('start')} to {meta.get('temporal_range', {}).get('end')}")
        print(f"Total Days: {meta.get('temporal_range', {}).get('total_days')}")
        
    except Exception as e:
        print(f"\nCould not read metadata: {e}")
    
    return all_found

if __name__ == "__main__":
    verify_l4_files()