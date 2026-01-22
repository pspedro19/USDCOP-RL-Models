from minio import Minio
import os
import sys

# MinIO Config (from environment or defaults)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
SECURE = False

def clear_buckets():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=SECURE
    )

    buckets = ["forecasts", "ml-models"]
    
    for bucket_name in buckets:
        if client.bucket_exists(bucket_name):
            print(f"Clearing bucket: {bucket_name}")
            # List all objects
            objects = client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                client.remove_object(bucket_name, obj.object_name)
                print(f"  Deleted: {obj.object_name}")
        else:
            print(f"Bucket {bucket_name} does not exist.")

if __name__ == "__main__":
    try:
        clear_buckets()
        print("MinIO cleanup complete.")
    except Exception as e:
        print(f"Error clearing MinIO: {e}")
