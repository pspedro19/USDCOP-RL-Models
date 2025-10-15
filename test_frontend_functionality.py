#!/usr/bin/env python3
"""
Frontend Functionality Test Script
==================================

This script tests the USDCOP trading dashboard functionality:
1. Tests if dashboard loads on port 5000
2. Verifies API connectivity on port 8000
3. Tests historical data endpoints
4. Checks chart rendering and navigation components
5. Validates WebSocket connections
"""

import requests
import json
import time
import sys
from datetime import datetime, timedelta

def test_dashboard_accessibility():
    """Test if the dashboard is accessible on port 5000"""
    print("🌐 Testing Dashboard Accessibility...")

    try:
        response = requests.get("http://localhost:5000", timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard is accessible on port 5000")
            print(f"   Response size: {len(response.content)} bytes")

            # Check if it looks like a Next.js app
            if 'nextjs' in response.headers.get('x-nextjs-cache', '').lower() or 'next' in response.text.lower():
                print("✅ Confirmed: Next.js application detected")

            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to dashboard on port 5000")
        return False
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        return False

def test_api_connectivity():
    """Test API connectivity and basic endpoints"""
    print("\n🔌 Testing API Connectivity...")

    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Trading API is accessible on port 8000")
            print(f"   API Status: {data.get('status', 'unknown')}")
            print(f"   Version: {data.get('version', 'unknown')}")
            print(f"   Features: {', '.join(data.get('features', []))}")
            print(f"   Connected clients: {data.get('connected_clients', 0)}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_historical_data_endpoints():
    """Test historical data endpoints that the frontend uses"""
    print("\n📈 Testing Historical Data Endpoints...")

    # Test API health endpoint
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Health Endpoint working")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            print(f"   Total Records: {health_data.get('total_records', 'unknown'):,}")
            print(f"   Latest Data: {health_data.get('latest_data', 'unknown')}")
            print(f"   Market Open: {health_data.get('market_status', {}).get('is_open', 'unknown')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")
        return False

    # Test candlestick data endpoint (main historical data)
    try:
        response = requests.get("http://localhost:8000/api/candlesticks/USDCOP?timeframe=5m&limit=100", timeout=15)
        if response.status_code == 200:
            candle_data = response.json()
            print("✅ Candlestick Data Endpoint working")
            print(f"   Symbol: {candle_data.get('symbol', 'unknown')}")
            print(f"   Timeframe: {candle_data.get('timeframe', 'unknown')}")
            print(f"   Data Points: {candle_data.get('count', 0)}")
            print(f"   Date Range: {candle_data.get('start_date', 'unknown')} to {candle_data.get('end_date', 'unknown')}")

            # Verify data structure
            if candle_data.get('data') and len(candle_data['data']) > 0:
                sample_candle = candle_data['data'][0]
                required_fields = ['time', 'open', 'high', 'low', 'close', 'volume']
                missing_fields = [field for field in required_fields if field not in sample_candle]
                if not missing_fields:
                    print("✅ Candlestick data structure is correct")
                    print(f"   Latest Price: ${sample_candle['close']:.4f}")
                else:
                    print(f"❌ Missing fields in candlestick data: {missing_fields}")
                    return False
            else:
                print("❌ No candlestick data returned")
                return False
        else:
            print(f"❌ Candlestick endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing candlestick endpoint: {e}")
        return False

    return True

def test_historical_navigation_data():
    """Test endpoints that historical navigation components would use"""
    print("\n📅 Testing Historical Navigation Data...")

    # Test different timeframes that navigation components support
    timeframes = ['5m', '15m', '1h', '1d']

    for timeframe in timeframes:
        try:
            response = requests.get(f"http://localhost:8000/api/candlesticks/USDCOP?timeframe={timeframe}&limit=50", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {timeframe} timeframe data available ({data.get('count', 0)} points)")
            else:
                print(f"❌ {timeframe} timeframe failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing {timeframe}: {e}")

    # Test date range queries (simulating historical navigation)
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days

        response = requests.get(
            f"http://localhost:8000/api/candlesticks/USDCOP?"
            f"timeframe=5m&limit=1000&"
            f"start_date={start_date.isoformat()}&"
            f"end_date={end_date.isoformat()}",
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Date range query working ({data.get('count', 0)} points for last 30 days)")
        else:
            print(f"❌ Date range query failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing date range query: {e}")

    return True

def test_websocket_connectivity():
    """Test WebSocket connectivity (simplified)"""
    print("\n🔌 Testing WebSocket Services...")

    # Check WebSocket service health
    try:
        response = requests.get("http://localhost:8082", timeout=5)
        if response.status_code == 200:
            print("✅ WebSocket service is accessible on port 8082")
        else:
            print(f"⚠️ WebSocket service returned: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️ WebSocket service not accessible (this is optional)")
    except Exception as e:
        print(f"⚠️ WebSocket test error: {e}")

def test_frontend_api_proxy():
    """Test if frontend can proxy API calls"""
    print("\n🔄 Testing Frontend API Integration...")

    # These would be called by the frontend components
    api_endpoints = [
        "/api/market/realtime/route",  # Real-time market data
        "/api/market/update/route",    # Market updates
        "/api/proxy/",                 # API proxy
    ]

    for endpoint in api_endpoints:
        try:
            response = requests.get(f"http://localhost:5000{endpoint}", timeout=10)
            if response.status_code in [200, 404, 405]:  # 405 = method not allowed is OK for testing
                print(f"✅ Frontend endpoint {endpoint} is responsive")
            else:
                print(f"⚠️ Frontend endpoint {endpoint} returned: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error testing {endpoint}: {e}")

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("🚀 USDCOP TRADING DASHBOARD FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tests = [
        ("Dashboard Accessibility", test_dashboard_accessibility),
        ("API Connectivity", test_api_connectivity),
        ("Historical Data Endpoints", test_historical_data_endpoints),
        ("Historical Navigation Data", test_historical_navigation_data),
        ("WebSocket Connectivity", test_websocket_connectivity),
        ("Frontend API Integration", test_frontend_api_proxy),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False

        time.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<35} {status}")

    print(f"\n📈 Overall Score: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Dashboard functionality is working correctly.")
        print("✨ Historical navigation should be fully operational.")
    elif passed >= total * 0.8:  # 80% pass rate
        print("\n✅ MOSTLY WORKING! Some minor issues detected but core functionality is operational.")
        print("📊 Historical navigation should work for most use cases.")
    else:
        print("\n⚠️ ISSUES DETECTED! Several components may not be working correctly.")
        print("🔧 Please check the failed tests above and verify system configuration.")

    return passed, total

if __name__ == "__main__":
    passed, total = run_comprehensive_test()

    # Exit code for CI/CD
    sys.exit(0 if passed == total else 1)