import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
print("Sys path:")
for p in sys.path:
    print(f"  {p}")

try:
    import src
    print(f"\nImported src from: {src.__file__}")
    from src.features.common import prepare_features
    print("Imported prepare_features successfully")
except Exception as e:
    print(f"\nERROR Importing: {e}")
    import traceback
    traceback.print_exc()
