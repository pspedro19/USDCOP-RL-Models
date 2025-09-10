#!/usr/bin/env python3
"""
Check MinIO data storage and update L0 bucket with latest data
"""

from minio import Minio
from datetime import datetime, timedelta
import json
import os

# MinIO configuration
minio_client = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin123',
    secure=False
)

def check_buckets():
    """List all buckets and their contents"""
    print("=" * 60)
    print("CHECKING MINIO BUCKETS")
    print("=" * 60)
    
    buckets = minio_client.list_buckets()
    
    for bucket in buckets:
        print(f"\n[BUCKET] {bucket.name}")
        print(f"   Created: {bucket.creation_date}")
        
        # Count objects and get latest
        objects = list(minio_client.list_objects(bucket.name, recursive=True))
        
        if objects:
            print(f"   Objects: {len(objects)}")
            
            # Get latest 5 files
            latest_objects = sorted(objects, key=lambda x: x.last_modified, reverse=True)[:5]
            print("   Latest files:")
            for obj in latest_objects:
                print(f"     - {obj.object_name} ({obj.size} bytes) - {obj.last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("   Objects: 0 (Empty)")

def check_realtime_data():
    """Check realtime data bucket"""
    print("\n" + "=" * 60)
    print("CHECKING REALTIME DATA")
    print("=" * 60)
    
    bucket_name = 'realtime-usdcop-data'
    
    # Check if bucket exists
    if not minio_client.bucket_exists(bucket_name):
        print(f"[ERROR] Bucket '{bucket_name}' does not exist!")
        print("Creating bucket...")
        minio_client.make_bucket(bucket_name)
        print("[OK] Bucket created")
        return
    
    # List recent data
    objects = list(minio_client.list_objects(bucket_name, recursive=True))
    
    if objects:
        print(f"[OK] Found {len(objects)} files in realtime bucket")
        
        # Get today's data
        today = datetime.now().strftime('%Y-%m-%d')
        today_files = [obj for obj in objects if today in obj.object_name]
        
        if today_files:
            print(f"[DATA] Today's files ({today}): {len(today_files)}")
            for obj in today_files[:5]:
                print(f"   - {obj.object_name}")
        else:
            print(f"[WARNING] No data found for today ({today})")
    else:
        print("[WARNING] Realtime bucket is empty")

def check_l0_data():
    """Check L0 raw data bucket"""
    print("\n" + "=" * 60)
    print("CHECKING L0 RAW DATA")
    print("=" * 60)
    
    bucket_name = '00-raw-usdcop-marketdata'
    
    # Check if bucket exists
    if not minio_client.bucket_exists(bucket_name):
        print(f"[ERROR] L0 bucket '{bucket_name}' does not exist!")
        print("Creating bucket...")
        minio_client.make_bucket(bucket_name)
        print("[OK] Bucket created")
        return
    
    # List recent data
    objects = list(minio_client.list_objects(bucket_name, recursive=True))
    
    if objects:
        print(f"[OK] Found {len(objects)} files in L0 bucket")
        
        # Get latest files
        latest_objects = sorted(objects, key=lambda x: x.last_modified, reverse=True)[:10]
        
        print("\n[FILES] Latest L0 files:")
        for obj in latest_objects:
            date_str = obj.last_modified.strftime('%Y-%m-%d %H:%M:%S')
            size_kb = obj.size / 1024
            print(f"   - {obj.object_name[:50]}... ({size_kb:.1f} KB) - {date_str}")
        
        # Check for recent updates
        now = datetime.now()
        recent_threshold = now - timedelta(hours=1)
        recent_files = [obj for obj in objects if obj.last_modified.replace(tzinfo=None) > recent_threshold.replace(tzinfo=None)]
        
        if recent_files:
            print(f"\n[UPDATE] Files updated in last hour: {len(recent_files)}")
        else:
            print("\n[WARNING] No updates in the last hour")
    else:
        print("[WARNING] L0 bucket is empty")

def main():
    try:
        check_buckets()
        check_realtime_data()
        check_l0_data()
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print("1. If realtime bucket is empty, click 'Align Dataset' in dashboard")
        print("2. Data should be saved every 5 minutes during market hours")
        print("3. L0 bucket should be updated with aligned data")
        print("\n[COMPLETE] Check complete!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()