#!/usr/bin/env python3
"""
Comprehensive Pipeline Connectivity Verification
Tests all pipeline levels L0-L6, data sources, and APIs
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
import requests
from minio import Minio
from minio.error import S3Error

# Configuration
MINIO_CONFIG = {
    'endpoint': 'localhost:9000',
    'access_key': 'minioadmin',
    'secret_key': 'minioadmin123',
    'secure': False
}

TWELVEDATA_API_KEY = "9d7c480871f54f66bb933d96d5837d28"

# Pipeline bucket mapping
BUCKET_MAPPING = {
    'L0': '00-raw-usdcop-marketdata',
    'L1': '01-l1-ds-usdcop-standardize', 
    'L2': '02-l2-ds-usdcop-prepare',
    'L3': '03-l3-ds-usdcop-feature',
    'L4': '04-l4-ds-usdcop-rlready',
    'L5': '05-l5-ds-usdcop-serving',
    'L6': '99-common-trading-reports'
}

class ConnectivityTester:
    def __init__(self):
        self.minio_client = None
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'tests': {},
            'summary': {},
            'recommendations': []
        }
        
    def initialize_minio(self) -> bool:
        """Initialize MinIO client"""
        try:
            self.minio_client = Minio(**MINIO_CONFIG)
            # Test connection
            buckets = list(self.minio_client.list_buckets())
            print(f"✅ MinIO connected - Found {len(buckets)} buckets")
            return True
        except Exception as e:
            print(f"❌ MinIO connection failed: {e}")
            return False
    
    def test_docker_services(self) -> Dict[str, Any]:
        """Test Docker services availability"""
        print("\n🐳 Testing Docker Services...")
        
        services = {
            'minio': 'http://localhost:9000/minio/health/live',
            'postgres': 'postgresql://localhost:5432',  # Will test with pg_isready
            'redis': 'redis://localhost:6379',
            'airflow': 'http://localhost:8081/health'
        }
        
        results = {}
        
        # Test MinIO
        try:
            response = requests.get(services['minio'], timeout=5)
            results['minio'] = {
                'status': 'HEALTHY' if response.status_code == 200 else 'ERROR',
                'response_time': response.elapsed.total_seconds(),
                'error': None
            }
            print(f"  ✅ MinIO: {results['minio']['status']}")
        except Exception as e:
            results['minio'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ❌ MinIO: ERROR - {e}")
        
        # Test Airflow
        try:
            response = requests.get(services['airflow'], timeout=5)
            results['airflow'] = {
                'status': 'HEALTHY' if response.status_code == 200 else 'ERROR',
                'response_time': response.elapsed.total_seconds(),
                'error': None
            }
            print(f"  ✅ Airflow: {results['airflow']['status']}")
        except Exception as e:
            results['airflow'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ❌ Airflow: ERROR - {e}")
        
        return results
    
    async def test_twelvedata_api(self) -> Dict[str, Any]:
        """Test TwelveData API connectivity and endpoints"""
        print("\n📡 Testing TwelveData API...")
        
        endpoints = [
            ('quote', f'https://api.twelvedata.com/quote?symbol=USD/COP&apikey={TWELVEDATA_API_KEY}'),
            ('time_series', f'https://api.twelvedata.com/time_series?symbol=USD/COP&interval=5min&outputsize=10&apikey={TWELVEDATA_API_KEY}'),
            ('rsi', f'https://api.twelvedata.com/rsi?symbol=USD/COP&interval=5min&time_period=14&apikey={TWELVEDATA_API_KEY}')
        ]
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint_name, url in endpoints:
                try:
                    start_time = time.time()
                    async with session.get(url, timeout=10) as response:
                        response_time = time.time() - start_time
                        data = await response.json()
                        
                        if response.status == 200 and 'status' not in data:
                            results[endpoint_name] = {
                                'status': 'HEALTHY',
                                'response_time': response_time,
                                'data_keys': list(data.keys()) if isinstance(data, dict) else None,
                                'error': None
                            }
                            print(f"  ✅ {endpoint_name}: HEALTHY ({response_time:.2f}s)")
                        else:
                            error_msg = data.get('message', 'Unknown error') if isinstance(data, dict) else 'Invalid response'
                            results[endpoint_name] = {
                                'status': 'ERROR',
                                'response_time': response_time,
                                'error': error_msg
                            }
                            print(f"  ❌ {endpoint_name}: ERROR - {error_msg}")
                            
                except Exception as e:
                    results[endpoint_name] = {
                        'status': 'ERROR',
                        'response_time': None,
                        'error': str(e)
                    }
                    print(f"  ❌ {endpoint_name}: ERROR - {e}")
        
        return results
    
    def test_pipeline_buckets(self) -> Dict[str, Any]:
        """Test all pipeline buckets L0-L6"""
        print("\n📊 Testing Pipeline Buckets...")
        
        if not self.minio_client:
            return {'error': 'MinIO client not initialized'}
        
        results = {}
        
        for level, bucket_name in BUCKET_MAPPING.items():
            try:
                # Check if bucket exists
                if not self.minio_client.bucket_exists(bucket_name):
                    results[level] = {
                        'status': 'ERROR',
                        'bucket': bucket_name,
                        'error': 'Bucket does not exist',
                        'file_count': 0,
                        'latest_file': None,
                        'data_freshness_hours': None
                    }
                    print(f"  ❌ {level} ({bucket_name}): Bucket missing")
                    continue
                
                # List objects
                objects = list(self.minio_client.list_objects(bucket_name, recursive=True))
                
                # Find latest file
                latest_file = None
                data_freshness_hours = None
                
                if objects:
                    latest_obj = max(objects, key=lambda x: x.last_modified)
                    latest_file = {
                        'name': latest_obj.object_name,
                        'size': latest_obj.size,
                        'last_modified': latest_obj.last_modified.isostring()
                    }
                    
                    # Calculate data freshness
                    time_diff = datetime.now() - latest_obj.last_modified.replace(tzinfo=None)
                    data_freshness_hours = time_diff.total_seconds() / 3600
                
                # Determine status based on data freshness
                if data_freshness_hours is None:
                    status = 'ERROR'
                elif data_freshness_hours < 2:
                    status = 'HEALTHY'
                elif data_freshness_hours < 24:
                    status = 'WARNING'
                else:
                    status = 'ERROR'
                
                results[level] = {
                    'status': status,
                    'bucket': bucket_name,
                    'file_count': len(objects),
                    'latest_file': latest_file,
                    'data_freshness_hours': data_freshness_hours,
                    'error': None
                }
                
                status_icon = '✅' if status == 'HEALTHY' else '⚠️' if status == 'WARNING' else '❌'
                freshness_str = f"({data_freshness_hours:.1f}h ago)" if data_freshness_hours else "(no data)"
                print(f"  {status_icon} {level} ({bucket_name}): {len(objects)} files {freshness_str}")
                
            except S3Error as e:
                results[level] = {
                    'status': 'ERROR',
                    'bucket': bucket_name,
                    'error': f'S3 Error: {e}',
                    'file_count': 0,
                    'latest_file': None,
                    'data_freshness_hours': None
                }
                print(f"  ❌ {level} ({bucket_name}): S3 Error - {e}")
            except Exception as e:
                results[level] = {
                    'status': 'ERROR',
                    'bucket': bucket_name,
                    'error': str(e),
                    'file_count': 0,
                    'latest_file': None,
                    'data_freshness_hours': None
                }
                print(f"  ❌ {level} ({bucket_name}): Error - {e}")
        
        return results
    
    def test_frontend_connectivity(self) -> Dict[str, Any]:
        """Test frontend dashboard connectivity"""
        print("\n🌐 Testing Frontend Dashboard...")
        
        dashboard_url = 'http://localhost:3000'
        
        try:
            response = requests.get(dashboard_url, timeout=10)
            
            if response.status_code == 200:
                result = {
                    'status': 'HEALTHY',
                    'response_time': response.elapsed.total_seconds(),
                    'error': None
                }
                print(f"  ✅ Dashboard: HEALTHY ({result['response_time']:.2f}s)")
            else:
                result = {
                    'status': 'ERROR',
                    'response_time': response.elapsed.total_seconds(),
                    'error': f'HTTP {response.status_code}'
                }
                print(f"  ❌ Dashboard: ERROR - HTTP {response.status_code}")
                
        except Exception as e:
            result = {
                'status': 'ERROR',
                'response_time': None,
                'error': str(e)
            }
            print(f"  ❌ Dashboard: ERROR - {e}")
        
        return result
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check pipeline data freshness
        pipeline_results = self.results['tests'].get('pipeline_buckets', {})
        for level, result in pipeline_results.items():
            if result['status'] == 'ERROR':
                if 'does not exist' in str(result.get('error', '')):
                    recommendations.append(f"Create missing {level} bucket: {result['bucket']}")
                elif result['data_freshness_hours'] and result['data_freshness_hours'] > 24:
                    recommendations.append(f"Update {level} pipeline - data is {result['data_freshness_hours']:.1f} hours old")
                elif result['file_count'] == 0:
                    recommendations.append(f"Generate initial data for {level} pipeline")
        
        # Check API connectivity
        api_results = self.results['tests'].get('twelvedata_api', {})
        failed_endpoints = [name for name, result in api_results.items() if result['status'] == 'ERROR']
        if failed_endpoints:
            recommendations.append(f"Fix TwelveData API connectivity for: {', '.join(failed_endpoints)}")
        
        # Check Docker services
        docker_results = self.results['tests'].get('docker_services', {})
        failed_services = [name for name, result in docker_results.items() if result['status'] == 'ERROR']
        if failed_services:
            recommendations.append(f"Restart Docker services: {', '.join(failed_services)}")
        
        # Check frontend
        frontend_result = self.results['tests'].get('frontend_dashboard', {})
        if frontend_result.get('status') == 'ERROR':
            recommendations.append("Start frontend dashboard: npm run dev")
        
        if not recommendations:
            recommendations.append("All systems operational - no immediate actions required")
        
        return recommendations
    
    def calculate_overall_status(self) -> str:
        """Calculate overall system status"""
        all_tests = []
        
        for test_category, results in self.results['tests'].items():
            if isinstance(results, dict):
                if 'status' in results:  # Single test
                    all_tests.append(results['status'])
                else:  # Multiple tests
                    for result in results.values():
                        if isinstance(result, dict) and 'status' in result:
                            all_tests.append(result['status'])
        
        if not all_tests:
            return 'UNKNOWN'
        
        error_count = all_tests.count('ERROR')
        warning_count = all_tests.count('WARNING')
        healthy_count = all_tests.count('HEALTHY')
        
        total_tests = len(all_tests)
        
        if error_count == 0 and warning_count == 0:
            return 'HEALTHY'
        elif error_count / total_tests < 0.3:
            return 'WARNING'
        else:
            return 'ERROR'
    
    async def run_all_tests(self):
        """Run all connectivity tests"""
        print("🔍 USDCOP Trading RL - Pipeline Connectivity Verification")
        print("=" * 60)
        
        # Initialize MinIO
        if not self.initialize_minio():
            self.results['tests']['minio_init'] = {'status': 'ERROR', 'error': 'Failed to initialize MinIO'}
            return
        
        # Run tests
        self.results['tests']['docker_services'] = self.test_docker_services()
        self.results['tests']['twelvedata_api'] = await self.test_twelvedata_api()
        self.results['tests']['pipeline_buckets'] = self.test_pipeline_buckets()
        self.results['tests']['frontend_dashboard'] = self.test_frontend_connectivity()
        
        # Generate summary
        self.results['overall_status'] = self.calculate_overall_status()
        self.results['recommendations'] = self.generate_recommendations()
        
        # Count summary
        pipeline_healthy = sum(1 for r in self.results['tests']['pipeline_buckets'].values() 
                              if r.get('status') == 'HEALTHY')
        pipeline_total = len(self.results['tests']['pipeline_buckets'])
        
        api_healthy = sum(1 for r in self.results['tests']['twelvedata_api'].values() 
                         if r.get('status') == 'HEALTHY')
        api_total = len(self.results['tests']['twelvedata_api'])
        
        self.results['summary'] = {
            'pipeline_connectivity': f"{pipeline_healthy}/{pipeline_total} levels healthy",
            'api_connectivity': f"{api_healthy}/{api_total} endpoints healthy",
            'total_tests_run': sum(len(v) if isinstance(v, dict) else 1 for v in self.results['tests'].values()),
            'overall_health_score': pipeline_healthy / pipeline_total if pipeline_total > 0 else 0
        }
        
        # Print results
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 CONNECTIVITY TEST SUMMARY")
        print("=" * 60)
        
        status_icon = {
            'HEALTHY': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'UNKNOWN': '❓'
        }
        
        print(f"Overall Status: {status_icon.get(self.results['overall_status'], '❓')} {self.results['overall_status']}")
        print(f"Pipeline Connectivity: {self.results['summary']['pipeline_connectivity']}")
        print(f"API Connectivity: {self.results['summary']['api_connectivity']}")
        print(f"Health Score: {self.results['summary']['overall_health_score']:.1%}")
        
        if self.results['recommendations']:
            print(f"\n📋 RECOMMENDATIONS:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"\nTest completed at: {self.results['timestamp']}")
        print("=" * 60)
    
    def save_results(self):
        """Save test results to file"""
        filename = f"connectivity_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"C:\\Users\\pedro\\OneDrive\\Documents\\ALGO TRADING\\USDCOP\\USDCOP_Trading_RL\\{filename}"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📁 Results saved to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

async def main():
    """Main execution function"""
    tester = ConnectivityTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)