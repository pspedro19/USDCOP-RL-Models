#!/usr/bin/env python3
"""
Performance Testing Suite for USDCOP Trading System
"""

import requests
import time
import threading
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class PerformanceTester:
    def __init__(self):
        self.results = []
        self.errors = []

    def test_api_endpoint(self, url, test_name, timeout=30):
        """Test a single API endpoint"""
        start_time = time.time()
        try:
            response = requests.get(url, timeout=timeout)
            end_time = time.time()

            result = {
                'test_name': test_name,
                'url': url,
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'success': response.status_code == 200,
                'timestamp': datetime.now().isoformat(),
                'response_size': len(response.content)
            }

            if response.status_code == 200:
                try:
                    result['response_data'] = response.json()
                except:
                    result['response_data'] = response.text[:100]

            return result

        except Exception as e:
            end_time = time.time()
            return {
                'test_name': test_name,
                'url': url,
                'status_code': 0,
                'response_time': end_time - start_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def concurrent_load_test(self, url, num_requests=10, num_threads=3):
        """Test API under concurrent load"""
        print(f"🔥 Load testing {url} with {num_requests} requests using {num_threads} threads")

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.test_api_endpoint, url, f"load_test_{i}")
                for i in range(num_requests)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['success']:
                    print(f"✅ Request completed in {result['response_time']:.3f}s")
                else:
                    print(f"❌ Request failed: {result.get('error', 'Unknown error')}")

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate statistics
        successful_requests = [r for r in results if r['success']]
        success_rate = len(successful_requests) / len(results) * 100

        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            median_response_time = statistics.median(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = median_response_time = 0

        throughput = len(successful_requests) / total_time if total_time > 0 else 0

        load_test_summary = {
            'url': url,
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(results) - len(successful_requests),
            'success_rate': success_rate,
            'total_time': total_time,
            'throughput': throughput,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'median_response_time': median_response_time
        }

        print(f"📊 Load Test Results for {url}:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Throughput: {throughput:.2f} requests/second")
        print(f"   Avg Response Time: {avg_response_time:.3f}s")
        print(f"   Min/Max Response Time: {min_response_time:.3f}s / {max_response_time:.3f}s")

        return load_test_summary

    def test_database_performance(self):
        """Test database performance through API"""
        print("\n💾 Testing Database Performance")
        print("=" * 50)

        # Test health check (simple query)
        health_result = self.test_api_endpoint("http://localhost:8000/api/health", "db_health_check")

        if health_result['success']:
            health_data = health_result['response_data']
            print(f"✅ Database Health Check: {health_result['response_time']:.3f}s")
            print(f"   Total Records: {health_data.get('total_records', 'N/A')}")
            print(f"   Latest Data: {health_data.get('latest_data', 'N/A')}")
        else:
            print(f"❌ Database Health Check Failed: {health_result.get('error', 'Unknown')}")

        return health_result

    def test_memory_usage(self):
        """Test memory usage of containers"""
        print("\n🧠 Testing Memory Usage")
        print("=" * 30)

        try:
            import subprocess
            result = subprocess.run(['docker', 'stats', '--no-stream', '--format',
                                   'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                print("📊 Container Resource Usage:")
                print(result.stdout)
            else:
                print(f"❌ Failed to get container stats: {result.stderr}")
        except Exception as e:
            print(f"❌ Memory test failed: {e}")

def main():
    print("🧪 USDCOP Performance Testing Suite")
    print("=" * 60)

    tester = PerformanceTester()

    # Test 1: Basic API performance
    print("\n🔸 Test 1: Basic API Endpoint Performance")
    endpoints = [
        ("http://localhost:8000/api/health", "Health Check"),
        ("http://localhost:8000/api/market-status", "Market Status"),
        ("http://localhost:8085/health", "Orchestrator Health"),
        ("http://localhost:8082/health", "WebSocket Health"),
        ("http://localhost:3000/api/health", "Dashboard Health")
    ]

    for url, name in endpoints:
        result = tester.test_api_endpoint(url, name)
        if result['success']:
            print(f"✅ {name}: {result['response_time']:.3f}s")
        else:
            print(f"❌ {name}: {result.get('error', 'Failed')}")

    # Test 2: Database performance
    tester.test_database_performance()

    # Test 3: Load testing
    print("\n🔸 Test 3: Load Testing")

    # Load test health endpoint (lightweight)
    health_load_result = tester.concurrent_load_test(
        "http://localhost:8000/api/health",
        num_requests=20,
        num_threads=5
    )

    # Load test market status endpoint
    market_status_load_result = tester.concurrent_load_test(
        "http://localhost:8000/api/market-status",
        num_requests=15,
        num_threads=3
    )

    # Test 4: Memory usage
    tester.test_memory_usage()

    # Test 5: WebSocket capacity test
    print("\n🔸 Test 5: WebSocket Service Capacity")
    websocket_result = tester.test_api_endpoint("http://localhost:8082/health", "WebSocket Capacity")
    if websocket_result['success']:
        ws_data = websocket_result['response_data']
        print(f"✅ WebSocket Service: {websocket_result['response_time']:.3f}s")
        print(f"   Active Connections: {ws_data.get('connections', 'N/A')}")
        print(f"   Redis Connected: {ws_data.get('redis_connected', 'N/A')}")

    # Summary
    print("\n📋 Performance Test Summary")
    print("=" * 40)
    print(f"🔹 Health Check Load Test:")
    print(f"   Success Rate: {health_load_result['success_rate']:.1f}%")
    print(f"   Throughput: {health_load_result['throughput']:.2f} req/s")
    print(f"   Avg Response: {health_load_result['avg_response_time']:.3f}s")

    print(f"🔹 Market Status Load Test:")
    print(f"   Success Rate: {market_status_load_result['success_rate']:.1f}%")
    print(f"   Throughput: {market_status_load_result['throughput']:.2f} req/s")
    print(f"   Avg Response: {market_status_load_result['avg_response_time']:.3f}s")

    print("\n✅ Performance testing completed!")

if __name__ == "__main__":
    main()