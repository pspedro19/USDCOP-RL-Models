#!/usr/bin/env python3
"""
Test timezone handling and market hours functionality
"""

import requests
import json
from datetime import datetime, timezone
import pytz

def test_market_hours_api():
    """Test the market hours API endpoints"""
    print("🕐 Testing Market Hours and Timezone Functionality")
    print("=" * 60)

    # Test market status endpoint
    try:
        response = requests.get("http://localhost:8000/api/market-status", timeout=10)
        if response.status_code == 200:
            market_data = response.json()
            print("✅ Market Status API working:")
            print(f"   Is Open: {market_data.get('is_open')}")
            print(f"   Current Time: {market_data.get('current_time')}")
            print(f"   Timezone: {market_data.get('timezone')}")
            print(f"   Trading Hours: {market_data.get('trading_hours')}")
            print(f"   Next Event: {market_data.get('next_event')} ({market_data.get('next_event_type')})")
            print(f"   Time to Next Event: {market_data.get('time_to_next_event')}")
        else:
            print(f"❌ Market Status API failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Market Status API error: {e}")

    print("\n" + "-" * 40)

    # Test health endpoint for market status
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            market_status = health_data.get('market_status', {})
            print("✅ Health API Market Status:")
            print(f"   Database Records: {health_data.get('total_records', 'N/A')}")
            print(f"   Latest Data: {health_data.get('latest_data', 'N/A')}")
            print(f"   Real-time Monitor: {health_data.get('real_time_monitor', 'N/A')}")
            print(f"   Market Open: {market_status.get('is_open', 'N/A')}")
            print(f"   Current COT Time: {market_status.get('current_time', 'N/A')}")
        else:
            print(f"❌ Health API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health API error: {e}")

def test_timezone_calculations():
    """Test timezone calculations manually"""
    print("\n🌍 Testing Timezone Calculations")
    print("=" * 40)

    # Get current time in different timezones
    utc_now = datetime.now(timezone.utc)
    cot_tz = pytz.timezone('America/Bogota')
    cot_now = utc_now.astimezone(cot_tz)

    print(f"UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"COT Time: {cot_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Check if it's a trading day (Monday-Friday)
    is_weekday = cot_now.weekday() < 5  # 0=Monday, 6=Sunday
    print(f"Is Weekday: {is_weekday}")

    # Check trading hours (8:00 AM - 12:55 PM COT)
    market_start = cot_now.replace(hour=8, minute=0, second=0, microsecond=0)
    market_end = cot_now.replace(hour=12, minute=55, second=0, microsecond=0)

    is_market_hours = market_start <= cot_now <= market_end
    is_market_open = is_weekday and is_market_hours

    print(f"Market Start: {market_start.strftime('%H:%M:%S')}")
    print(f"Market End: {market_end.strftime('%H:%M:%S')}")
    print(f"Current Time: {cot_now.strftime('%H:%M:%S')}")
    print(f"Is Market Hours: {is_market_hours}")
    print(f"Is Market Open: {is_market_open}")

    # Calculate next market event
    if is_market_open:
        next_event = market_end
        next_event_type = "market_close"
    elif is_weekday and cot_now < market_start:
        next_event = market_start
        next_event_type = "market_open"
    else:
        # Calculate next Monday 8 AM
        days_until_monday = (7 - cot_now.weekday()) % 7
        if days_until_monday == 0:  # Today is Monday but after hours
            days_until_monday = 7
        next_monday = cot_now.replace(hour=8, minute=0, second=0, microsecond=0) + \
                     pytz.timezone('America/Bogota').localize(datetime(1970, 1, 1)).utctimetuple()
        next_event = cot_now + \
                    pytz.timezone('America/Bogota').localize(datetime.combine(
                        (cot_now + pytz.timezone('America/Bogota').localize(datetime(1970, 1, 1)).utctimetuple()).date() +
                        pytz.timezone('America/Bogota').localize(datetime(1970, 1, 1)).utctimetuple(),
                        datetime.min.time()
                    )).replace(tzinfo=None)
        next_event_type = "market_open"

    print(f"Next Event: {next_event_type}")

def test_realtime_orchestrator():
    """Test the real-time orchestrator status"""
    print("\n🚀 Testing Real-time Orchestrator")
    print("=" * 40)

    try:
        response = requests.get("http://localhost:8085/health", timeout=10)
        if response.status_code == 200:
            orchestrator_data = response.json()
            print("✅ Realtime Orchestrator Status:")
            print(f"   Service: {orchestrator_data.get('service')}")
            print(f"   Version: {orchestrator_data.get('version')}")
            print(f"   L0 Pipeline Completed: {orchestrator_data.get('l0_pipeline_completed')}")
            print(f"   Realtime Collecting: {orchestrator_data.get('realtime_collecting')}")

            market_session = orchestrator_data.get('market_session', {})
            print(f"   Session Date: {market_session.get('session_date')}")
            print(f"   Session Open: {market_session.get('is_open')}")
            print(f"   Current Time: {market_session.get('current_time')}")
        else:
            print(f"❌ Orchestrator API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Orchestrator API error: {e}")

if __name__ == "__main__":
    print("🧪 USDCOP Timezone and Market Hours Testing")
    print("=" * 60)

    test_market_hours_api()
    test_timezone_calculations()
    test_realtime_orchestrator()

    print("\n✅ Timezone testing completed!")