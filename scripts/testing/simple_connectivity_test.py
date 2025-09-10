#!/usr/bin/env python3
"""
Simple Pipeline Connectivity Test
Tests all pipeline levels L0-L6, data sources, and APIs
"""

import json
import sys
import time
from datetime import datetime
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

def test_minio_connectivity():
    """Test MinIO connectivity and bucket status"""
    print("\n[TESTING] MinIO Connectivity and Pipeline Buckets")
    print("-" * 50)
    
    try:
        # Initialize MinIO client
        minio_client = Minio(**MINIO_CONFIG)
        
        # Test connection
        buckets = list(minio_client.list_buckets())
        print(f"[OK] MinIO connected - Found {len(buckets)} buckets")
        
        # Test each pipeline bucket
        results = {}
        for level, bucket_name in BUCKET_MAPPING.items():
            try:
                if not minio_client.bucket_exists(bucket_name):
                    print(f"[ERROR] {level} ({bucket_name}): Bucket missing")
                    results[level] = {'status': 'ERROR', 'error': 'Bucket missing'}
                    continue
                
                # List objects
                objects = list(minio_client.list_objects(bucket_name, recursive=True))
                
                if objects:
                    latest_obj = max(objects, key=lambda x: x.last_modified)
                    time_diff = datetime.now() - latest_obj.last_modified.replace(tzinfo=None)
                    hours_ago = time_diff.total_seconds() / 3600
                    
                    status = 'HEALTHY' if hours_ago < 2 else 'WARNING' if hours_ago < 24 else 'ERROR'
                    print(f"[{status}] {level} ({bucket_name}): {len(objects)} files, latest {hours_ago:.1f}h ago")
                    results[level] = {'status': status, 'files': len(objects), 'hours_ago': hours_ago}
                else:
                    print(f"[WARNING] {level} ({bucket_name}): Empty bucket")
                    results[level] = {'status': 'WARNING', 'files': 0}
                    
            except Exception as e:
                print(f"[ERROR] {level} ({bucket_name}): {e}")
                results[level] = {'status': 'ERROR', 'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"[ERROR] MinIO connection failed: {e}")
        return {'error': str(e)}

def test_twelvedata_api():
    """Test TwelveData API connectivity"""
    print("\n[TESTING] TwelveData API Connectivity")
    print("-" * 50)
    
    endpoints = [
        ('quote', f'https://api.twelvedata.com/quote?symbol=USD/COP&apikey={TWELVEDATA_API_KEY}'),
        ('time_series', f'https://api.twelvedata.com/time_series?symbol=USD/COP&interval=5min&outputsize=5&apikey={TWELVEDATA_API_KEY}'),
    ]
    
    results = {}
    
    for endpoint_name, url in endpoints:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if 'status' not in data or data.get('status') != 'error':
                    print(f"[OK] {endpoint_name}: Response in {response_time:.2f}s")
                    results[endpoint_name] = {'status': 'HEALTHY', 'response_time': response_time}
                else:
                    error_msg = data.get('message', 'Unknown error')
                    print(f"[ERROR] {endpoint_name}: {error_msg}")
                    results[endpoint_name] = {'status': 'ERROR', 'error': error_msg}
            else:
                print(f"[ERROR] {endpoint_name}: HTTP {response.status_code}")
                results[endpoint_name] = {'status': 'ERROR', 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"[ERROR] {endpoint_name}: {e}")
            results[endpoint_name] = {'status': 'ERROR', 'error': str(e)}
    
    return results

def test_docker_services():
    """Test Docker services"""
    print("\n[TESTING] Docker Services")
    print("-" * 50)
    
    services = {
        'MinIO': 'http://localhost:9000/minio/health/live',
        'Airflow': 'http://localhost:8081/health'
    }
    
    results = {}
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"[OK] {service_name}: Healthy")
                results[service_name] = {'status': 'HEALTHY'}
            else:
                print(f"[ERROR] {service_name}: HTTP {response.status_code}")
                results[service_name] = {'status': 'ERROR', 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            print(f"[ERROR] {service_name}: {e}")
            results[service_name] = {'status': 'ERROR', 'error': str(e)}
    
    return results

def test_frontend_dashboard():
    """Test frontend dashboard"""
    print("\n[TESTING] Frontend Dashboard")
    print("-" * 50)
    
    try:
        response = requests.get('http://localhost:3000', timeout=10)
        if response.status_code == 200:
            print("[OK] Dashboard: Accessible")
            return {'status': 'HEALTHY'}
        else:
            print(f"[ERROR] Dashboard: HTTP {response.status_code}")
            return {'status': 'ERROR', 'error': f'HTTP {response.status_code}'}
    except Exception as e:
        print(f"[ERROR] Dashboard: {e}")
        return {'status': 'ERROR', 'error': str(e)}

def generate_summary(all_results):
    """Generate test summary"""
    print("\n" + "=" * 60)
    print("PIPELINE CONNECTIVITY TEST SUMMARY")
    print("=" * 60)
    
    # Count statuses
    total_tests = 0
    healthy_tests = 0
    warning_tests = 0
    error_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            if 'status' in results:  # Single test
                total_tests += 1
                if results['status'] == 'HEALTHY':
                    healthy_tests += 1
                elif results['status'] == 'WARNING':
                    warning_tests += 1
                elif results['status'] == 'ERROR':
                    error_tests += 1
            else:  # Multiple tests
                for result in results.values():
                    if isinstance(result, dict) and 'status' in result:
                        total_tests += 1
                        if result['status'] == 'HEALTHY':
                            healthy_tests += 1
                        elif result['status'] == 'WARNING':
                            warning_tests += 1
                        elif result['status'] == 'ERROR':
                            error_tests += 1
    
    # Calculate overall status
    if error_tests == 0 and warning_tests == 0:
        overall_status = 'HEALTHY'
    elif error_tests / total_tests < 0.3:
        overall_status = 'WARNING'  
    else:
        overall_status = 'ERROR'
    
    print(f"Overall Status: {overall_status}")
    print(f"Tests Run: {total_tests}")
    print(f"Healthy: {healthy_tests}")
    print(f"Warning: {warning_tests}")
    print(f"Error: {error_tests}")
    
    if total_tests > 0:
        health_score = healthy_tests / total_tests
        print(f"Health Score: {health_score:.1%}")
    
    # Pipeline specific summary
    if 'minio_buckets' in all_results and isinstance(all_results['minio_buckets'], dict):
        pipeline_levels = len([k for k in all_results['minio_buckets'].keys() if k.startswith('L')])
        pipeline_healthy = len([v for k, v in all_results['minio_buckets'].items() 
                               if k.startswith('L') and v.get('status') == 'HEALTHY'])
        print(f"Pipeline Levels: {pipeline_healthy}/{pipeline_levels} healthy")
    
    # Recommendations
    print(f"\nRecommendations:")
    recommendations = []
    
    if 'minio_buckets' in all_results:
        for level, result in all_results['minio_buckets'].items():
            if result.get('status') == 'ERROR':
                recommendations.append(f"Fix {level} pipeline bucket issues")
            elif result.get('status') == 'WARNING' and result.get('files', 0) == 0:
                recommendations.append(f"Generate initial data for {level} pipeline")
    
    if 'api_tests' in all_results:
        api_errors = [k for k, v in all_results['api_tests'].items() if v.get('status') == 'ERROR']
        if api_errors:
            recommendations.append(f"Fix API connectivity: {', '.join(api_errors)}")
    
    if 'docker_services' in all_results:
        service_errors = [k for k, v in all_results['docker_services'].items() if v.get('status') == 'ERROR']
        if service_errors:
            recommendations.append(f"Restart Docker services: {', '.join(service_errors)}")
    
    if not recommendations:
        recommendations.append("All systems operational")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return {
        'overall_status': overall_status,
        'total_tests': total_tests,
        'healthy_tests': healthy_tests,
        'warning_tests': warning_tests,
        'error_tests': error_tests,
        'health_score': health_score if total_tests > 0 else 0,
        'recommendations': recommendations
    }

def main():
    """Main test execution"""
    print("USDCOP Trading RL - Pipeline Connectivity Verification")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    
    # Run all tests
    all_results = {
        'minio_buckets': test_minio_connectivity(),
        'api_tests': test_twelvedata_api(),
        'docker_services': test_docker_services(),
        'frontend_dashboard': test_frontend_dashboard()
    }
    
    # Generate summary
    summary = generate_summary(all_results)
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'detailed_results': all_results
    }
    
    try:
        filename = f"connectivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {filename}")
    except Exception as e:
        print(f"Failed to save report: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)