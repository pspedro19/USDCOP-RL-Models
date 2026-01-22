#!/usr/bin/env python3
"""
Test script for trading calendar validation in the API
Tests the holiday/weekend validation endpoints
"""

import requests
import json
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_is_trading_day():
    """Test the is-trading-day endpoint"""
    print("\n" + "="*60)
    print("Testing /api/v1/trading-calendar/is-trading-day endpoint")
    print("="*60)

    # Test today
    print("\n1. Testing today's date:")
    response = requests.get(f"{BASE_URL}/api/v1/trading-calendar/is-trading-day")
    if response.status_code == 200:
        data = response.json()
        print(f"   Date: {data['date']}")
        print(f"   Is Trading Day: {data['is_trading_day']}")
        print(f"   Is Weekend: {data['is_weekend']}")
        print(f"   Is Holiday: {data['is_holiday']}")
        if data.get('reason'):
            print(f"   Reason: {data['reason']}")
        if data.get('next_trading_day'):
            print(f"   Next Trading Day: {data['next_trading_day']}")
    else:
        print(f"   ERROR: {response.status_code} - {response.text}")

    # Test specific dates
    test_dates = [
        "2025-12-25",  # Christmas (holiday)
        "2025-12-20",  # Saturday (weekend)
        "2025-12-21",  # Sunday (weekend)
        "2025-01-01",  # New Year (holiday)
        "2025-07-20",  # Independence Day (holiday)
        "2025-12-17",  # Regular Tuesday (should be trading day)
    ]

    print("\n2. Testing specific dates:")
    for test_date in test_dates:
        response = requests.get(f"{BASE_URL}/api/v1/trading-calendar/is-trading-day?date={test_date}")
        if response.status_code == 200:
            data = response.json()
            status = "✓ TRADING DAY" if data['is_trading_day'] else "✗ NOT TRADING"
            reason = f" ({data.get('reason', 'N/A')})" if not data['is_trading_day'] else ""
            print(f"   {test_date}: {status}{reason}")
        else:
            print(f"   {test_date}: ERROR - {response.status_code}")

def test_inference_endpoint():
    """Test the inference endpoint with holiday validation"""
    print("\n" + "="*60)
    print("Testing /api/v1/inference endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/api/v1/inference")

    print(f"\nResponse Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Message: {data['message']}")

        if data['status'] == 'market_closed':
            print(f"Reason: {data.get('reason', 'N/A')}")
            print(f"Current Date: {data.get('current_date', 'N/A')}")
            print(f"Next Trading Day: {data.get('next_trading_day', 'N/A')}")
            print(f"Is Weekend: {data.get('is_weekend', 'N/A')}")
            print(f"Is Holiday: {data.get('is_holiday', 'N/A')}")
    else:
        print(f"ERROR: {response.text}")

def test_latest_price_endpoint():
    """Test the latest price endpoint with holiday validation"""
    print("\n" + "="*60)
    print("Testing /api/latest/USDCOP endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/api/latest/USDCOP")

    print(f"\nResponse Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Symbol: {data['symbol']}")
        print(f"Price: {data['price']}")
        print(f"Timestamp: {data['timestamp']}")
        print(f"Source: {data['source']}")
    elif response.status_code == 425:
        # Market closed
        detail = response.json().get('detail', {})
        print(f"Market Status: CLOSED")
        print(f"Error: {detail.get('error', 'N/A')}")
        print(f"Message: {detail.get('message', 'N/A')}")
        if detail.get('current_date'):
            print(f"Current Date: {detail['current_date']}")
            print(f"Next Trading Day: {detail.get('next_trading_day', 'N/A')}")
            print(f"Is Weekend: {detail.get('is_weekend', 'N/A')}")
            print(f"Is Holiday: {detail.get('is_holiday', 'N/A')}")
    else:
        print(f"ERROR: {response.text}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TRADING CALENDAR API VALIDATION TEST")
    print("="*60)
    print(f"API Base URL: {BASE_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Test health endpoint first
        print("\nChecking API health...")
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is healthy and responsive")
        else:
            print(f"⚠ API health check returned: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR: Cannot connect to API at {BASE_URL}")
        print(f"  Make sure the API is running: python services/trading_api_realtime.py")
        print(f"  Error: {e}")
        return

    # Run tests
    test_is_trading_day()
    test_inference_endpoint()
    test_latest_price_endpoint()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
