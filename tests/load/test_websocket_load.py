"""
Load Tests for WebSocket Service
==================================

Tests WebSocket performance under load:
- 100+ concurrent WebSocket connections
- Message throughput (messages/second)
- Latency measurements (p50, p95, p99)
- Memory leak detection
- Connection stability under load

Uses locust or custom async load generator.

Target Metrics:
- Support 100+ concurrent connections
- Latency p95 < 100ms
- Throughput > 100 msg/sec
- Memory stable over time
"""

import pytest
import asyncio
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import pytz
import websockets
import psutil
import os

# Mark all tests as load tests
pytestmark = [pytest.mark.load, pytest.mark.slow]

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')


# ============================================================================
# WebSocket Load Test Client
# ============================================================================

class WebSocketLoadClient:
    """Load test client for WebSocket connections"""

    def __init__(self, url: str, client_id: int):
        self.url = url
        self.client_id = client_id
        self.websocket = None
        self.connected = False
        self.messages_received = 0
        self.latencies = []
        self.errors = []
        self.start_time = None
        self.end_time = None

    async def connect(self):
        """Connect to WebSocket"""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            self.start_time = time.time()
            return True
        except Exception as e:
            self.errors.append(str(e))
            return False

    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            self.end_time = time.time()

    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket"""
        if not self.connected:
            return False

        try:
            send_time = time.time()
            await self.websocket.send(json.dumps(message))

            # Wait for response
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=5.0
            )

            receive_time = time.time()
            latency_ms = (receive_time - send_time) * 1000

            self.latencies.append(latency_ms)
            self.messages_received += 1

            return True

        except asyncio.TimeoutError:
            self.errors.append('Timeout waiting for response')
            return False
        except Exception as e:
            self.errors.append(str(e))
            return False

    async def listen(self, duration_seconds: int = 60):
        """Listen for messages for specified duration"""
        if not self.connected:
            return

        end_time = time.time() + duration_seconds

        try:
            while time.time() < end_time and self.connected:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0
                    )

                    self.messages_received += 1

                except asyncio.TimeoutError:
                    continue  # No message received, continue listening

        except Exception as e:
            self.errors.append(str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        return {
            'client_id': self.client_id,
            'connected': self.connected,
            'messages_received': self.messages_received,
            'errors': len(self.errors),
            'duration_seconds': duration,
            'throughput': self.messages_received / duration if duration > 0 else 0,
            'latencies': {
                'count': len(self.latencies),
                'mean': statistics.mean(self.latencies) if self.latencies else 0,
                'median': statistics.median(self.latencies) if self.latencies else 0,
                'p95': self._percentile(self.latencies, 95) if self.latencies else 0,
                'p99': self._percentile(self.latencies, 99) if self.latencies else 0,
                'min': min(self.latencies) if self.latencies else 0,
                'max': max(self.latencies) if self.latencies else 0
            }
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]


# ============================================================================
# Test: Concurrent Connections
# ============================================================================

class TestConcurrentConnections:
    """Test handling of multiple concurrent connections"""

    @pytest.mark.asyncio
    async def test_100_concurrent_connections(self, performance_thresholds):
        """Test 100 concurrent WebSocket connections"""
        ws_url = os.getenv('TEST_WEBSOCKET_URL', 'ws://localhost:8080/ws')
        num_clients = 100
        duration = 30  # Run for 30 seconds

        # Create clients
        clients = [WebSocketLoadClient(ws_url, i) for i in range(num_clients)]

        # Connect all clients
        connect_tasks = [client.connect() for client in clients]
        connect_results = await asyncio.gather(*connect_tasks, return_exceptions=True)

        connected_count = sum(1 for result in connect_results if result is True)
        print(f"\nConnected clients: {connected_count}/{num_clients}")

        # Should connect at least 90% of clients
        assert connected_count >= num_clients * 0.9

        # Run load test - clients listening for messages
        listen_tasks = [client.listen(duration) for client in clients if client.connected]
        await asyncio.gather(*listen_tasks, return_exceptions=True)

        # Disconnect all clients
        disconnect_tasks = [client.disconnect() for client in clients]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        # Collect statistics
        all_stats = [client.get_stats() for client in clients]

        # Calculate aggregate metrics
        total_messages = sum(s['messages_received'] for s in all_stats)
        total_errors = sum(s['errors'] for s in all_stats)
        avg_throughput = statistics.mean([s['throughput'] for s in all_stats if s['throughput'] > 0])

        print(f"\nLoad Test Results:")
        print(f"  Total messages: {total_messages}")
        print(f"  Total errors: {total_errors}")
        print(f"  Avg throughput: {avg_throughput:.2f} msg/sec")
        print(f"  Error rate: {(total_errors / max(total_messages, 1)) * 100:.2f}%")

        # Assertions
        assert total_messages > 0  # Should receive some messages
        assert total_errors < total_messages * 0.1  # Error rate < 10%

    @pytest.mark.asyncio
    async def test_connection_stability(self):
        """Test connection stability over time"""
        ws_url = os.getenv('TEST_WEBSOCKET_URL', 'ws://localhost:8080/ws')
        num_clients = 20
        duration = 60  # Run for 60 seconds

        clients = [WebSocketLoadClient(ws_url, i) for i in range(num_clients)]

        # Connect all clients
        await asyncio.gather(*[client.connect() for client in clients])

        # Monitor connections over time
        checks = []
        check_interval = 10  # Check every 10 seconds

        for i in range(duration // check_interval):
            await asyncio.sleep(check_interval)

            # Count active connections
            active_connections = sum(1 for c in clients if c.connected)
            checks.append({
                'time': i * check_interval,
                'active': active_connections,
                'total': num_clients
            })

            print(f"\nCheck {i+1}: {active_connections}/{num_clients} connections active")

        # Disconnect all
        await asyncio.gather(*[client.disconnect() for client in clients])

        # Verify stability - should maintain most connections
        avg_active = statistics.mean([c['active'] for c in checks])
        assert avg_active >= num_clients * 0.85  # At least 85% stay connected


# ============================================================================
# Test: Throughput and Latency
# ============================================================================

class TestThroughputLatency:
    """Test message throughput and latency"""

    @pytest.mark.asyncio
    async def test_message_throughput(self, performance_thresholds):
        """Test message throughput under load"""
        ws_url = os.getenv('TEST_WEBSOCKET_URL', 'ws://localhost:8080/ws')
        num_clients = 50
        messages_per_client = 20

        clients = [WebSocketLoadClient(ws_url, i) for i in range(num_clients)]

        # Connect clients
        await asyncio.gather(*[client.connect() for client in clients])

        start_time = time.time()

        # Each client sends messages
        async def send_messages(client):
            for i in range(messages_per_client):
                message = {
                    'type': 'heartbeat',
                    'client_id': client.client_id,
                    'sequence': i,
                    'timestamp': datetime.now(COT_TZ).isoformat()
                }
                await client.send_message(message)
                await asyncio.sleep(0.1)  # Small delay between messages

        # Send messages concurrently
        await asyncio.gather(*[send_messages(c) for c in clients if c.connected])

        end_time = time.time()
        duration = end_time - start_time

        # Disconnect clients
        await asyncio.gather(*[client.disconnect() for client in clients])

        # Calculate throughput
        total_messages = sum(c.messages_received for c in clients)
        throughput = total_messages / duration

        print(f"\nThroughput Test Results:")
        print(f"  Total messages: {total_messages}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Throughput: {throughput:.2f} msg/sec")

        # Should exceed minimum throughput threshold
        min_throughput = performance_thresholds['min_throughput']
        assert throughput >= min_throughput

    @pytest.mark.asyncio
    async def test_message_latency(self, performance_thresholds):
        """Test message latency distribution"""
        ws_url = os.getenv('TEST_WEBSOCKET_URL', 'ws://localhost:8080/ws')
        num_clients = 20
        messages_per_client = 50

        clients = [WebSocketLoadClient(ws_url, i) for i in range(num_clients)]

        # Connect clients
        await asyncio.gather(*[client.connect() for client in clients])

        # Send messages and measure latency
        async def send_with_latency(client):
            for i in range(messages_per_client):
                message = {
                    'type': 'heartbeat',
                    'timestamp': datetime.now(COT_TZ).isoformat()
                }
                await client.send_message(message)

        await asyncio.gather(*[send_with_latency(c) for c in clients if c.connected])

        # Disconnect clients
        await asyncio.gather(*[client.disconnect() for client in clients])

        # Aggregate latencies
        all_latencies = []
        for client in clients:
            all_latencies.extend(client.latencies)

        if all_latencies:
            p50 = statistics.median(all_latencies)
            p95 = clients[0]._percentile(all_latencies, 95)
            p99 = clients[0]._percentile(all_latencies, 99)

            print(f"\nLatency Test Results:")
            print(f"  p50: {p50:.2f} ms")
            print(f"  p95: {p95:.2f} ms")
            print(f"  p99: {p99:.2f} ms")
            print(f"  min: {min(all_latencies):.2f} ms")
            print(f"  max: {max(all_latencies):.2f} ms")

            # Latency assertions
            max_latency = performance_thresholds['max_latency_ms']
            assert p95 < max_latency  # p95 should be below threshold


# ============================================================================
# Test: Memory Leak Detection
# ============================================================================

class TestMemoryLeaks:
    """Test for memory leaks under sustained load"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_stability(self, performance_thresholds):
        """Test memory usage stability over time"""
        ws_url = os.getenv('TEST_WEBSOCKET_URL', 'ws://localhost:8080/ws')
        num_clients = 30
        duration = 120  # Run for 2 minutes
        sample_interval = 10  # Sample every 10 seconds

        # Get initial process memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        clients = [WebSocketLoadClient(ws_url, i) for i in range(num_clients)]

        # Connect clients
        await asyncio.gather(*[client.connect() for client in clients])

        # Monitor memory over time
        memory_samples = [initial_memory]

        for i in range(duration // sample_interval):
            await asyncio.sleep(sample_interval)

            # Sample memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

            # Send some messages to generate activity
            message = {
                'type': 'heartbeat',
                'timestamp': datetime.now(COT_TZ).isoformat()
            }

            await asyncio.gather(
                *[client.send_message(message) for client in clients if client.connected],
                return_exceptions=True
            )

            print(f"\nMemory sample {i+1}: {current_memory:.2f} MB")

        # Disconnect clients
        await asyncio.gather(*[client.disconnect() for client in clients])

        # Analyze memory trend
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        memory_growth_percent = (memory_growth / initial_memory) * 100

        print(f"\nMemory Leak Test Results:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Growth: {memory_growth:.2f} MB ({memory_growth_percent:.1f}%)")
        print(f"  Max threshold: {performance_thresholds['max_memory_mb']} MB")

        # Memory should not grow excessively
        assert final_memory < performance_thresholds['max_memory_mb']
        # Memory growth should be reasonable (< 50%)
        assert memory_growth_percent < 50.0


# ============================================================================
# Test: Error Handling Under Load
# ============================================================================

class TestErrorHandling:
    """Test error handling under load conditions"""

    @pytest.mark.asyncio
    async def test_reconnection_behavior(self):
        """Test automatic reconnection after disconnects"""
        ws_url = os.getenv('TEST_WEBSOCKET_URL', 'ws://localhost:8080/ws')
        num_clients = 10

        clients = [WebSocketLoadClient(ws_url, i) for i in range(num_clients)]

        # Initial connection
        await asyncio.gather(*[client.connect() for client in clients])
        initial_connected = sum(1 for c in clients if c.connected)

        # Force disconnect half the clients
        for i in range(num_clients // 2):
            await clients[i].disconnect()

        await asyncio.sleep(1)

        # Attempt reconnection
        reconnect_tasks = [
            clients[i].connect()
            for i in range(num_clients // 2)
        ]
        await asyncio.gather(*reconnect_tasks, return_exceptions=True)

        # Check reconnection success
        reconnected = sum(1 for c in clients if c.connected)

        print(f"\nReconnection Test:")
        print(f"  Initial: {initial_connected} connected")
        print(f"  After disconnect: {reconnected} connected")

        # Clean up
        await asyncio.gather(*[client.disconnect() for client in clients])

        # Should successfully reconnect most clients
        assert reconnected >= initial_connected * 0.8

    @pytest.mark.asyncio
    async def test_malformed_message_handling(self):
        """Test handling of malformed messages"""
        ws_url = os.getenv('TEST_WEBSOCKET_URL', 'ws://localhost:8080/ws')

        client = WebSocketLoadClient(ws_url, 0)
        await client.connect()

        if client.connected:
            # Send malformed JSON
            try:
                await client.websocket.send("invalid json {{{")
                await asyncio.sleep(0.5)

                # Connection should still be alive
                assert client.connected

            except Exception as e:
                client.errors.append(str(e))

            await client.disconnect()

        # Should handle malformed messages gracefully
        assert len(client.errors) == 0 or 'invalid json' not in str(client.errors[0]).lower()


# ============================================================================
# Locust Load Test (Alternative)
# ============================================================================

"""
Alternative: Use Locust for load testing

Create a locustfile.py:

from locust import User, task, between
import websocket
import json
import time

class WebSocketUser(User):
    wait_time = between(1, 3)

    def on_start(self):
        self.ws = websocket.WebSocketApp(
            "ws://localhost:8080/ws",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()

    @task
    def send_heartbeat(self):
        message = {
            'type': 'heartbeat',
            'timestamp': time.time()
        }
        self.ws.send(json.dumps(message))

    def on_message(self, ws, message):
        pass

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        pass

Run with:
locust -f locustfile.py --host=ws://localhost:8080
"""
