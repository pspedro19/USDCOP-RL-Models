#!/usr/bin/env python3
"""
Trading Signals Service - Test Suite
=====================================
Quick test script to verify service functionality.

Usage:
    python test_service.py
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8003"
WS_URL = "ws://localhost:8003/ws/signals"


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_root():
    """Test root endpoint"""
    print_section("Test 1: Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_health():
    """Test health check"""
    print_section("Test 2: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_detailed_health():
    """Test detailed health check"""
    print_section("Test 3: Detailed Health Check")
    try:
        response = requests.get(f"{BASE_URL}/api/signals/health")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        print("\nHealth Summary:")
        print(f"  Service Status: {data.get('status', 'unknown')}")
        print(f"  Model Loaded: {data.get('model_loaded', False)}")
        print(f"  Database Connected: {data.get('database_connected', False)}")
        print(f"  Uptime: {data.get('uptime_seconds', 0):.1f}s")
        print(f"  Total Signals: {data.get('total_signals_generated', 0)}")

        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_model_info():
    """Test model information endpoint"""
    print_section("Test 4: Model Information")
    try:
        response = requests.get(f"{BASE_URL}/api/signals/model/info")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_generate_signal():
    """Test signal generation"""
    print_section("Test 5: Generate Signal")
    try:
        payload = {
            "symbol": "USDCOP",
            "close_price": 4250.50,
            "open_price": 4245.00,
            "high_price": 4255.00,
            "low_price": 4240.00,
            "volume": 1500000,
            "rsi": 35.5,
            "macd": -2.3,
            "macd_signal": -1.8
        }

        print(f"Request payload: {json.dumps(payload, indent=2)}")

        response = requests.post(
            f"{BASE_URL}/api/signals/generate",
            json=payload
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            signal = data.get('signal', {})

            print("\nGenerated Signal:")
            print(f"  Signal ID: {signal.get('signal_id', 'N/A')}")
            print(f"  Action: {signal.get('action', 'N/A')}")
            print(f"  Confidence: {signal.get('confidence', 0):.2%}")
            print(f"  Entry Price: {signal.get('entry_price', 0):.2f}")
            print(f"  Stop Loss: {signal.get('stop_loss', 0):.2f}")
            print(f"  Take Profit: {signal.get('take_profit', 0):.2f}")
            print(f"  Risk/Reward: {signal.get('risk_reward_ratio', 0):.2f}")
            print(f"  Position Size: {signal.get('position_size', 0):.2%}")
            print(f"  Latency: {signal.get('latency_ms', 0):.2f}ms")

            if 'reasoning' in signal:
                print(f"\n  Reasoning:")
                for reason in signal['reasoning']:
                    print(f"    - {reason}")

            return True
        else:
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_latest_signal():
    """Test get latest signal"""
    print_section("Test 6: Get Latest Signal")
    try:
        response = requests.get(f"{BASE_URL}/api/signals/latest")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            signal = data.get('signal', {})
            print(f"\nLatest Signal:")
            print(f"  Action: {signal.get('action', 'N/A')}")
            print(f"  Confidence: {signal.get('confidence', 0):.2%}")
            print(f"  Timestamp: {signal.get('timestamp', 'N/A')}")
            return True
        elif response.status_code == 404:
            print("No signals available yet (expected if first run)")
            return True
        else:
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_signal_history():
    """Test signal history"""
    print_section("Test 7: Signal History")
    try:
        response = requests.get(f"{BASE_URL}/api/signals/history?limit=10")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nSignal History:")
            print(f"  Total Signals: {data.get('count', 0)}")

            signals = data.get('signals', [])
            for i, signal in enumerate(signals[:3], 1):
                print(f"\n  Signal {i}:")
                print(f"    Action: {signal.get('action', 'N/A')}")
                print(f"    Confidence: {signal.get('confidence', 0):.2%}")
                print(f"    Timestamp: {signal.get('timestamp', 'N/A')}")

            if len(signals) > 3:
                print(f"\n  ... and {len(signals) - 3} more")

            return True
        else:
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_statistics():
    """Test statistics endpoint"""
    print_section("Test 8: Statistics")
    try:
        response = requests.get(f"{BASE_URL}/api/signals/statistics")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})

            print("\nService Statistics:")

            # Inference stats
            inference = stats.get('inference', {})
            print(f"\n  Inference:")
            print(f"    Total Inferences: {inference.get('total_inferences', 0)}")
            print(f"    Avg Latency: {inference.get('avg_latency_ms', 0):.2f}ms")

            # Signal stats
            signals = stats.get('signals', {})
            print(f"\n  Signals:")
            print(f"    Signals Generated: {signals.get('signals_generated', 0)}")

            # Position stats
            positions = stats.get('positions', {})
            print(f"\n  Positions:")
            print(f"    Total Positions: {positions.get('total_positions_opened', 0)}")
            print(f"    Active Positions: {positions.get('active_positions', 0)}")
            print(f"    Closed Positions: {positions.get('total_positions_closed', 0)}")
            print(f"    Win Rate: {positions.get('win_rate', 0):.1f}%")
            print(f"    Total PnL: {positions.get('total_pnl', 0):.2f}")

            return True
        else:
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_websocket():
    """Test WebSocket connection (basic)"""
    print_section("Test 9: WebSocket Connection")
    try:
        import websocket

        print("Testing WebSocket connection...")
        print("(This will just verify connection, not full functionality)")

        ws = websocket.create_connection(WS_URL)

        # Receive welcome message
        result = ws.recv()
        data = json.loads(result)
        print(f"\nReceived: {data.get('type', 'unknown')}")

        # Send ping
        ws.send(json.dumps({"type": "ping"}))

        # Receive pong
        result = ws.recv()
        data = json.loads(result)
        print(f"Ping response: {data.get('type', 'unknown')}")

        ws.close()
        print("\n✅ WebSocket connection successful")
        return True

    except ImportError:
        print("\n⚠️  websocket-client not installed")
        print("   Install with: pip install websocket-client")
        return True  # Don't fail the test
    except Exception as e:
        print(f"\n❌ WebSocket Error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print("  Trading Signals Service - Test Suite")
    print("=" * 60)
    print(f"  Target: {BASE_URL}")
    print(f"  Time: {datetime.utcnow().isoformat()}")
    print("=" * 60)

    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Detailed Health", test_detailed_health),
        ("Model Info", test_model_info),
        ("Generate Signal", test_generate_signal),
        ("Latest Signal", test_latest_signal),
        ("Signal History", test_signal_history),
        ("Statistics", test_statistics),
        ("WebSocket", test_websocket),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            time.sleep(0.5)  # Brief pause between tests
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys

    print("\nMake sure the service is running on http://localhost:8003")
    print("Press Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled")
        sys.exit(0)

    exit_code = run_all_tests()
    sys.exit(exit_code)
