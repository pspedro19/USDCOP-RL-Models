#!/usr/bin/env python3
"""
Test API connectivity through different endpoints
"""
import requests
import json

def test_endpoints():
    endpoints = [
        ("Direct API", "http://localhost:8000/api/latest/USDCOP"),
        ("Direct API Health", "http://localhost:8000/api/market/health"),
        ("Proxy API", "http://localhost:5000/api/proxy/trading/latest/USDCOP"),
        ("Proxy Health", "http://localhost:5000/api/proxy/trading/market/health"),
    ]

    print("Testing API connectivity...")
    print("=" * 60)

    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: SUCCESS")
                data = response.json()
                if 'price' in data:
                    print(f"   Price: {data['price']}")
                if 'status' in data:
                    print(f"   Status: {data['status']}")
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                print(f"   Response: {response.text[:100]}")
        except Exception as e:
            print(f"❌ {name}: FAILED")
            print(f"   Error: {str(e)}")
        print("-" * 60)

    print("\n🔧 Debugging Information:")
    print("1. API Server is running on port 8000")
    print("2. Dashboard is running on port 5000")
    print("3. Proxy should forward /api/proxy/trading/* to localhost:8000/api/*")

if __name__ == "__main__":
    test_endpoints()