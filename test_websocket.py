#!/usr/bin/env python3
"""
WebSocket Test Client for USDCOP Trading System
Test real-time data streaming capabilities
"""

import asyncio
import websockets
import json
import signal
import time
from datetime import datetime

class WebSocketTester:
    def __init__(self, uri="ws://localhost:8000/ws"):
        self.uri = uri
        self.connected = False
        self.messages_received = 0
        self.start_time = None

    async def test_connection(self):
        """Test WebSocket connection and message handling"""
        print(f"🔌 Connecting to WebSocket: {self.uri}")

        try:
            async with websockets.connect(self.uri) as websocket:
                self.connected = True
                self.start_time = time.time()
                print(f"✅ Connected successfully at {datetime.now()}")

                # Listen for messages
                timeout_seconds = 30
                print(f"📡 Listening for messages (timeout: {timeout_seconds}s)...")

                try:
                    while True:
                        try:
                            # Wait for message with timeout
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=timeout_seconds
                            )

                            self.messages_received += 1
                            data = json.loads(message)

                            print(f"📨 Message {self.messages_received} received:")
                            print(f"   Type: {data.get('type', 'unknown')}")
                            print(f"   Timestamp: {datetime.now()}")

                            if 'data' in data:
                                market_data = data['data']
                                print(f"   Symbol: {market_data.get('symbol', 'N/A')}")
                                print(f"   Price: {market_data.get('price', 'N/A')}")
                                print(f"   Data Timestamp: {market_data.get('timestamp', 'N/A')}")

                            print(f"   Raw: {json.dumps(data, indent=2)[:200]}...")
                            print("-" * 50)

                        except asyncio.TimeoutError:
                            print(f"⏰ No messages received in {timeout_seconds} seconds")
                            break

                except KeyboardInterrupt:
                    print("🛑 Connection interrupted by user")

        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except websockets.exceptions.InvalidURI:
            print("❌ Invalid WebSocket URI")
        except ConnectionRefusedError:
            print("❌ Connection refused - is the WebSocket service running?")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
        finally:
            self.connected = False
            duration = time.time() - self.start_time if self.start_time else 0
            print(f"\n📊 Test Summary:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Messages received: {self.messages_received}")
            print(f"   Average rate: {self.messages_received/duration:.2f} msg/s" if duration > 0 else "   Average rate: N/A")

async def test_multiple_connections():
    """Test multiple concurrent WebSocket connections"""
    print("\n🔀 Testing multiple concurrent connections...")

    async def single_connection(conn_id, duration=10):
        uri = f"ws://localhost:8000/ws"
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Connection {conn_id} established")
                start_time = time.time()
                message_count = 0

                while time.time() - start_time < duration:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        message_count += 1
                    except asyncio.TimeoutError:
                        continue

                print(f"📊 Connection {conn_id}: {message_count} messages in {duration}s")
                return message_count

        except Exception as e:
            print(f"❌ Connection {conn_id} failed: {e}")
            return 0

    # Test with 3 concurrent connections
    tasks = [single_connection(i+1, 15) for i in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_messages = sum(r for r in results if isinstance(r, int))
    print(f"📊 Total messages across all connections: {total_messages}")

async def main():
    print("🧪 USDCOP WebSocket Testing Suite")
    print("=" * 50)

    # Test 1: Basic connection
    print("\n🔸 Test 1: Basic WebSocket Connection")
    tester = WebSocketTester("ws://localhost:8000/ws")
    await tester.test_connection()

    # Test 2: Multiple connections
    await test_multiple_connections()

    print("\n✅ WebSocket testing completed!")

if __name__ == "__main__":
    asyncio.run(main())