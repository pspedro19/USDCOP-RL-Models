"""
Check what L4 data actually exists in MinIO
"""

from minio import Minio

client = Minio('localhost:9000', 'minioadmin', 'minioadmin', secure=False)

print("Checking L4 data in MinIO bucket 'ds-usdcop-rlready':")
print("-"*60)

objects = list(client.list_objects('ds-usdcop-rlready', recursive=True))

print(f"Total objects: {len(objects)}")

# Group by run_id
runs = {}
for obj in objects:
    if 'run_id=' in obj.object_name:
        parts = obj.object_name.split('run_id=')
        if len(parts) > 1:
            run_id = parts[1].split('/')[0]
            if run_id not in runs:
                runs[run_id] = []
            runs[run_id].append(obj.object_name)

print(f"\nFound {len(runs)} unique run IDs:")
for run_id, files in runs.items():
    print(f"\n  Run: {run_id}")
    print(f"  Files: {len(files)}")
    # Show first few files
    for f in files[:3]:
        print(f"    - {f.split('/')[-1]}")
    if len(files) > 3:
        print(f"    ... and {len(files)-3} more files")

# Check for L4_BACKFILL specifically
print("\n" + "="*60)
print("Searching for L4_BACKFILL runs:")
backfill_found = False
for run_id in runs.keys():
    if 'L4_BACKFILL' in run_id:
        print(f"  FOUND: {run_id}")
        backfill_found = True

if not backfill_found:
    print("  No L4_BACKFILL runs found in MinIO")
    print("\n  This means the 894-episode backfill may not have been saved properly.")
    print("  Action: Re-run the L4 backfill pipeline")