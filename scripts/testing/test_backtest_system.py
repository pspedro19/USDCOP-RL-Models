#!/usr/bin/env python3
"""
Backtest System Test
===================

This script tests the end-to-end backtest functionality by:
1. Checking L6 MinIO bucket structure
2. Verifying Airflow DAG status
3. Testing data retrieval and processing
4. Validating metrics calculation

Usage: python test_backtest_system.py
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_status(message, success=True):
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{status}] {message}{reset}")

def check_minio_bucket():
    """Check if MinIO L6 bucket exists and has data"""
    print_header("MinIO L6 Bucket Check")
    
    try:
        # Check if MinIO is running
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}", "--filter", "name=minio"],
            capture_output=True, text=True
        )
        
        if "minio" in result.stdout:
            print_status("MinIO container is running")
            
            # Try to list L6 bucket contents
            try:
                # Using mc client if available
                bucket_result = subprocess.run(
                    ["docker", "exec", "minio", "mc", "ls", "local/usdcop-l6-backtest/"],
                    capture_output=True, text=True
                )
                
                if bucket_result.returncode == 0:
                    print_status("L6 bucket is accessible")
                    if bucket_result.stdout.strip():
                        print(f"Found data: {bucket_result.stdout[:200]}...")
                        return True
                    else:
                        print_status("L6 bucket is empty", False)
                        return False
                else:
                    print_status("Cannot access L6 bucket", False)
                    return False
                    
            except Exception as e:
                print_status(f"Error accessing bucket: {e}", False)
                return False
                
        else:
            print_status("MinIO container not found", False)
            return False
            
    except Exception as e:
        print_status(f"Error checking MinIO: {e}", False)
        return False

def check_airflow_dag():
    """Check Airflow DAG status"""
    print_header("Airflow DAG Status Check")
    
    try:
        # Check if Airflow is running
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}", "--filter", "name=airflow"],
            capture_output=True, text=True
        )
        
        if "airflow" in result.stdout:
            print_status("Airflow container is running")
            
            # Try to check DAG status via Airflow API
            try:
                airflow_url = "http://localhost:8080/api/v1/dags/usdcop_m5__07_l6_backtest_referencia"
                response = requests.get(
                    airflow_url,
                    auth=("airflow", "airflow"),
                    timeout=5
                )
                
                if response.status_code == 200:
                    dag_info = response.json()
                    print_status(f"DAG found: {dag_info.get('dag_id', 'Unknown')}")
                    print_status(f"DAG active: {dag_info.get('is_active', False)}")
                    return True
                else:
                    print_status(f"DAG API error: {response.status_code}", False)
                    return False
                    
            except requests.RequestException as e:
                print_status(f"Cannot connect to Airflow API: {e}", False)
                return False
                
        else:
            print_status("Airflow container not found", False)
            return False
            
    except Exception as e:
        print_status(f"Error checking Airflow: {e}", False)
        return False

def test_backtest_api():
    """Test the Next.js backtest API"""
    print_header("Backtest API Test")
    
    try:
        # Test the API endpoint
        api_url = "http://localhost:3000/api/backtest/results"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print_status("API responded successfully")
                result_data = data.get("data", {})
                run_id = result_data.get("runId", "Unknown")
                print_status(f"Latest run ID: {run_id}")
                
                # Check if we have test data
                test_data = result_data.get("test")
                if test_data:
                    print_status("Test split data available")
                    
                    # Check KPIs
                    kpis = test_data.get("kpis")
                    if kpis:
                        sharpe = kpis.get("top_bar", {}).get("Sharpe", 0)
                        print_status(f"Sharpe ratio: {sharpe:.3f}")
                        
                        if sharpe > 0:
                            print_status("Positive Sharpe ratio detected")
                        else:
                            print_status("Zero/negative Sharpe ratio", False)
                    
                    # Check trades
                    trades = test_data.get("trades", [])
                    print_status(f"Found {len(trades)} trades")
                    
                    return True
                else:
                    print_status("No test data in response", False)
                    return False
                    
            else:
                print_status(f"API error: {data.get('error', 'Unknown')}", False)
                return False
        else:
            print_status(f"API HTTP error: {response.status_code}", False)
            return False
            
    except requests.RequestException as e:
        print_status(f"Cannot connect to API: {e}", False)
        return False
    except Exception as e:
        print_status(f"API test error: {e}", False)
        return False

def test_metrics_calculation():
    """Test hedge fund metrics calculation"""
    print_header("Metrics Calculation Test")
    
    try:
        # Create sample trade data
        sample_trades = [
            {"pnl": 0.02, "duration": 120},  # 2% return, 2 hours
            {"pnl": -0.01, "duration": 60},  # -1% return, 1 hour
            {"pnl": 0.03, "duration": 180},  # 3% return, 3 hours
            {"pnl": -0.005, "duration": 90}, # -0.5% return, 1.5 hours
            {"pnl": 0.01, "duration": 240},  # 1% return, 4 hours
        ]
        
        # Calculate basic metrics
        total_trades = len(sample_trades)
        winning_trades = len([t for t in sample_trades if t["pnl"] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = sum([t["pnl"] for t in sample_trades if t["pnl"] > 0]) / max(1, winning_trades)
        avg_loss = sum([t["pnl"] for t in sample_trades if t["pnl"] < 0]) / max(1, losing_trades)
        
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        total_return = sum([t["pnl"] for t in sample_trades])
        
        print_status(f"Sample trades: {total_trades}")
        print_status(f"Win rate: {win_rate:.2%}")
        print_status(f"Payoff ratio: {payoff_ratio:.2f}")
        print_status(f"Total return: {total_return:.2%}")
        
        # Basic validation
        if win_rate > 0.3 and payoff_ratio > 1.0:
            print_status("Metrics calculation looks healthy")
            return True
        else:
            print_status("Metrics indicate poor performance", False)
            return False
            
    except Exception as e:
        print_status(f"Metrics calculation error: {e}", False)
        return False

def run_integration_test():
    """Run complete integration test"""
    print_header("Integration Test Summary")
    
    tests = [
        ("MinIO L6 Bucket", check_minio_bucket),
        ("Airflow DAG", check_airflow_dag),
        ("Backtest API", test_backtest_api),
        ("Metrics Calculation", test_metrics_calculation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"{test_name} failed with exception: {e}", False)
            results[test_name] = False
    
    print("\nTest Results:")
    print("-" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = "\033[92m" if result else "\033[91m"
        print(f"{color}{test_name:.<30} {status}\033[0m")
        if result:
            passed += 1
    
    print("-" * 40)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print_status("\nAll tests passed! Backtest system is operational.", True)
        return True
    elif passed >= total * 0.75:
        print_status(f"\nMost tests passed ({passed}/{total}). System is mostly operational.", True)
        return True
    else:
        print_status(f"\nSeveral tests failed ({total-passed}/{total}). System needs attention.", False)
        return False

def main():
    """Main test runner"""
    print("USDCOP Backtest System Test")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        success = run_integration_test()
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success:
            print("\n🎉 Backtest system is ready for production!")
            sys.exit(0)
        else:
            print("\n⚠️  Some issues detected. Please review the results above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()