#!/usr/bin/env python3
"""
DAG Import Test Script for USDCOP RL Trading System
Tests all DAG files for import errors before deployment
"""

import os
import sys
import traceback
from pathlib import Path

def test_dag_imports():
    """Test all DAG files for import errors"""
    
    # Get the DAGs directory
    dags_dir = Path(__file__).parent / "dags"
    
    if not dags_dir.exists():
        print(f"❌ DAGs directory not found: {dags_dir}")
        return False
    
    print(f"🔍 Testing DAG imports in: {dags_dir}")
    print("=" * 60)
    
    # Find all Python files in the DAGs directory
    dag_files = list(dags_dir.glob("**/*.py"))
    
    if not dag_files:
        print("⚠️  No Python files found in DAGs directory")
        return True
    
    success_count = 0
    error_count = 0
    
    for dag_file in dag_files:
        # Skip __pycache__ and other non-DAG files
        if "__pycache__" in str(dag_file) or dag_file.name.startswith("__"):
            continue
            
        print(f"Testing: {dag_file.name}")
        
        try:
            # Get the relative path for import
            rel_path = dag_file.relative_to(dags_dir)
            module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
            
            # Add DAGs directory to Python path
            if str(dags_dir) not in sys.path:
                sys.path.insert(0, str(dags_dir))
            
            # Try to import the module
            __import__(module_name)
            
            print(f"  ✅ SUCCESS")
            success_count += 1
            
        except ImportError as e:
            print(f"  ❌ IMPORT ERROR: {e}")
            error_count += 1
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            print(f"  📝 Traceback:")
            traceback.print_exc()
            error_count += 1
    
    print("=" * 60)
    print(f"📊 Results: {success_count} successful, {error_count} errors")
    
    if error_count == 0:
        print("🎉 All DAG imports successful!")
        return True
    else:
        print(f"⚠️  {error_count} DAG files have import errors")
        return False

def test_critical_imports():
    """Test critical dependencies that are causing issues"""
    
    print("🧪 Testing critical dependencies...")
    print("=" * 60)
    
    critical_modules = [
        "scipy",
        "gymnasium", 
        "torch",
        "pandas",
        "numpy",
        "sklearn",
        "minio",
        "mlflow",
        "stable_baselines3"
    ]
    
    success_count = 0
    error_count = 0
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module} - FAILED: {e}")
            error_count += 1
        except Exception as e:
            print(f"❌ {module} - ERROR: {e}")
            error_count += 1
    
    print("=" * 60)
    print(f"📊 Dependencies: {success_count} available, {error_count} missing")
    
    return error_count == 0

def main():
    """Main test function"""
    
    print("🚀 USDCOP RL Trading System - DAG Import Test")
    print("=" * 60)
    
    # Test critical dependencies first
    deps_ok = test_critical_imports()
    print()
    
    # Test DAG imports
    dags_ok = test_dag_imports()
    
    print("\n" + "=" * 60)
    if deps_ok and dags_ok:
        print("🎉 ALL TESTS PASSED - Ready for deployment!")
        return 0
    else:
        print("❌ TESTS FAILED - Fix errors before deployment")
        if not deps_ok:
            print("   - Install missing dependencies")
        if not dags_ok:
            print("   - Fix DAG import errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())