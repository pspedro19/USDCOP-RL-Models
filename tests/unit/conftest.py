"""
Unit Tests conftest.py
Common fixtures and configuration for unit tests.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# COLLECT IGNORE: Skip files that cause hard crashes
# =============================================================================

def check_onnxruntime_available():
    """
    Check if onnxruntime can be imported without crashing.
    On Windows, onnxruntime can cause access violations.
    """
    try:
        # This is a lightweight check - don't actually import onnxruntime
        # which would cause the crash. Instead, check if the package exists.
        import importlib.util
        spec = importlib.util.find_spec("onnxruntime")
        if spec is None:
            return False

        # On Windows, even if the package exists, it may crash on import
        # Check platform and skip on Windows to avoid the access violation
        if sys.platform == "win32":
            return False  # Skip on Windows to avoid access violation

        return True
    except Exception:
        return False


# Files to skip if onnxruntime is not available
_onnxruntime_dependent_files = [
    "test_onnx_converter.py",
]

# Build collect_ignore list
collect_ignore = []

if not check_onnxruntime_available():
    collect_ignore.extend(_onnxruntime_dependent_files)
