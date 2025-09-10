#!/usr/bin/env python3
"""
Direct MinIO data analysis to count exact number of data points
Analyzes all historical data in MinIO buckets to verify 84,455 data points
"""

import json
import csv
from minio import Minio
import io
from datetime import datetime
import pandas as pd

# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"

def analyze_bucket_data(bucket_name):
    """Analyze all data files in a MinIO bucket"""
    print(f"\n=== ANALYZING BUCKET: {bucket_name} ===")
    
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        # List all objects in bucket
        objects = list(client.list_objects(bucket_name, recursive=True))
        print(f"Found {len(objects)} files in bucket")
        
        total_data_points = 0
        file_analysis = []
        date_range = {"earliest": None, "latest": None}
        
        for obj in objects:
            if not hasattr(obj, 'name') or not obj.name:
                continue
                
            file_points = 0
            file_start_date = None
            file_end_date = None
            
            try:
                # Get object data
                response = client.get_object(bucket_name, obj.name)
                content = response.read().decode('utf-8')
                
                print(f"\nAnalyzing: {obj.name} ({obj.size} bytes)")
                
                # Parse based on file type
                if obj.name.endswith('.csv'):
                    # Parse CSV file
                    lines = content.strip().split('\n')
                    if len(lines) > 1:  # Has header + data
                        file_points = len(lines) - 1  # Exclude header
                        
                        # Try to extract date range
                        try:
                            headers = lines[0].lower().split(',')
                            time_col = -1
                            
                            # Find time column
                            for i, header in enumerate(headers):
                                if any(keyword in header for keyword in ['time', 'date', 'datetime']):
                                    time_col = i
                                    break
                            
                            if time_col >= 0 and len(lines) > 2:
                                first_row = lines[1].split(',')
                                last_row = lines[-1].split(',')
                                
                                if len(first_row) > time_col:
                                    file_start_date = first_row[time_col].strip().strip('"')
                                if len(last_row) > time_col:
                                    file_end_date = last_row[time_col].strip().strip('"')
                        except Exception as e:
                            print(f"  Warning: Could not extract dates: {e}")
                
                elif obj.name.endswith('.json'):
                    # Parse JSON file
                    try:
                        data = json.loads(content)
                        
                        if isinstance(data, list):
                            file_points = len(data)
                            # Try to get date range from first/last items
                            if data:
                                if 'datetime' in data[0]:
                                    file_start_date = data[0]['datetime']
                                if 'datetime' in data[-1]:
                                    file_end_date = data[-1]['datetime']
                                    
                        elif isinstance(data, dict):
                            if 'data' in data and isinstance(data['data'], list):
                                file_points = len(data['data'])
                                # Get date range from nested data
                                if data['data'] and 'datetime' in data['data'][0]:
                                    file_start_date = data['data'][0]['datetime']
                                if data['data'] and 'datetime' in data['data'][-1]:
                                    file_end_date = data['data'][-1]['datetime']
                            else:
                                file_points = 1  # Single data object
                                if 'datetime' in data:
                                    file_start_date = file_end_date = data['datetime']
                                    
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Invalid JSON in {obj.name}: {e}")
                        continue
                
                # Update totals
                total_data_points += file_points
                
                # Track overall date range
                for date_str in [file_start_date, file_end_date]:
                    if date_str:
                        try:
                            # Try multiple date formats
                            date_obj = None
                            for fmt in ['%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ']:
                                try:
                                    date_obj = datetime.strptime(date_str.replace('Z', '+00:00').replace('+00:00', ''), fmt.replace('%z', ''))
                                    break
                                except:
                                    continue
                            
                            if date_obj:
                                if not date_range["earliest"] or date_obj < date_range["earliest"]:
                                    date_range["earliest"] = date_obj
                                if not date_range["latest"] or date_obj > date_range["latest"]:
                                    date_range["latest"] = date_obj
                        except:
                            pass
                
                file_analysis.append({
                    'file': obj.name,
                    'size_bytes': obj.size,
                    'data_points': file_points,
                    'start_date': file_start_date,
                    'end_date': file_end_date
                })
                
                print(f"  Data points: {file_points}")
                if file_start_date and file_end_date:
                    print(f"  Date range: {file_start_date} to {file_end_date}")
                
            except Exception as e:
                print(f"  Error processing {obj.name}: {e}")
                continue
        
        # Sort file analysis by data points (largest first)
        file_analysis.sort(key=lambda x: x['data_points'], reverse=True)
        
        print(f"\n=== BUCKET SUMMARY: {bucket_name} ===")
        print(f"Total Files: {len(file_analysis)}")
        print(f"Total Data Points: {total_data_points:,}")
        
        if date_range["earliest"] and date_range["latest"]:
            print(f"Date Range: {date_range['earliest'].strftime('%Y-%m-%d')} to {date_range['latest'].strftime('%Y-%m-%d')}")
            print(f"Time Span: {(date_range['latest'] - date_range['earliest']).days} days")
        
        # Show top files by data points
        print(f"\nTop 10 files by data points:")
        for i, file_info in enumerate(file_analysis[:10]):
            print(f"  {i+1:2}. {file_info['file']} - {file_info['data_points']:,} points")
        
        return total_data_points, date_range, file_analysis
        
    except Exception as e:
        print(f"Error analyzing bucket {bucket_name}: {e}")
        return 0, {}, []

def main():
    """Main analysis function"""
    print("="*80)
    print("MINIO DATA POINT VERIFICATION - DETAILED ANALYSIS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: 84,455 data points")
    
    # Buckets to analyze (in order of preference)
    buckets_to_check = [
        ("L1 Standardized", "01-l1-ds-usdcop-standardize"),
        ("L0 Raw Data", "00-raw-usdcop-marketdata"),
        ("L2 Prepared", "02-l2-ds-usdcop-prepare"),
        ("L3 Features", "03-l3-ds-usdcop-feature")
    ]
    
    total_across_buckets = 0
    all_results = {}
    
    for bucket_label, bucket_name in buckets_to_check:
        points, date_range, files = analyze_bucket_data(bucket_name)
        all_results[bucket_name] = {
            'points': points,
            'date_range': date_range,
            'files': files
        }
        total_across_buckets += points
    
    print("\n" + "="*80)
    print("OVERALL VERIFICATION SUMMARY")
    print("="*80)
    
    # Find the bucket with the most comprehensive data
    best_bucket = None
    max_points = 0
    
    for bucket_name, results in all_results.items():
        if results['points'] > max_points:
            max_points = results['points']
            best_bucket = bucket_name
    
    if best_bucket:
        print(f"\nMOST COMPREHENSIVE BUCKET: {best_bucket}")
        print(f"Data Points: {all_results[best_bucket]['points']:,}")
        
        date_range = all_results[best_bucket]['date_range']
        if date_range.get('earliest') and date_range.get('latest'):
            print(f"Date Range: {date_range['earliest'].strftime('%Y-%m-%d')} to {date_range['latest'].strftime('%Y-%m-%d')}")
            
            # Check if it covers 2020-2025
            covers_2020 = date_range['earliest'].year <= 2020
            covers_2025 = date_range['latest'].year >= 2024
            
            print(f"Covers 2020: {'YES' if covers_2020 else 'NO'}")
            print(f"Covers 2024+: {'YES' if covers_2025 else 'NO'}")
    
    # Compare with expected count
    expected_count = 84455
    if max_points > 0:
        deviation = abs(max_points - expected_count) / expected_count
        print(f"\nEXPECTED COUNT VERIFICATION:")
        print(f"Expected: {expected_count:,} points")
        print(f"Found: {max_points:,} points")
        print(f"Deviation: {deviation*100:.1f}%")
        
        if deviation < 0.05:  # Within 5%
            print("[EXCELLENT] Data count matches expected values")
        elif deviation < 0.15:  # Within 15%
            print("[GOOD] Data count is reasonably close to expected")
        else:
            print("[WARNING] Significant deviation from expected count")
    
    print(f"\nREPLAY FUNCTIONALITY ASSESSMENT:")
    if max_points >= 50000:
        print("[EXCELLENT] Sufficient data for comprehensive replay")
        print(f"  - Can support multiple speed settings (0.1x to 100x)")
        print(f"  - Covers extensive historical period")
        print(f"  - Suitable for professional trading analysis")
    elif max_points >= 10000:
        print("[GOOD] Adequate data for replay functionality")
        print(f"  - Supports basic replay features")
        print(f"  - Limited historical coverage")
    else:
        print("[LIMITED] Insufficient data for comprehensive replay")
    
    # Data quality assessment
    print(f"\nDATA QUALITY ASSESSMENT:")
    quality_score = 0
    
    if max_points > 80000:
        quality_score += 30
        print("✓ High data volume (30/30 points)")
    elif max_points > 50000:
        quality_score += 20
        print("✓ Good data volume (20/30 points)")
    else:
        quality_score += 10
        print("- Limited data volume (10/30 points)")
    
    if date_range.get('earliest') and date_range.get('latest'):
        span_days = (date_range['latest'] - date_range['earliest']).days
        if span_days > 1500:  # ~4 years
            quality_score += 30
            print("✓ Excellent time coverage (30/30 points)")
        elif span_days > 730:  # ~2 years
            quality_score += 20
            print("✓ Good time coverage (20/30 points)")
        else:
            quality_score += 10
            print("- Limited time coverage (10/30 points)")
    
    # File organization
    best_results = all_results.get(best_bucket, {})
    if best_results.get('files'):
        if len(best_results['files']) > 50:
            quality_score += 20
            print("✓ Well-organized file structure (20/20 points)")
        elif len(best_results['files']) > 10:
            quality_score += 15
            print("✓ Adequate file organization (15/20 points)")
        else:
            quality_score += 10
            print("- Basic file organization (10/20 points)")
    
    # Data gaps assessment
    if best_results.get('files'):
        # Estimate average points per file
        avg_points = best_results['points'] / len(best_results['files'])
        if avg_points > 1000:
            quality_score += 20
            print("✓ Dense data files, minimal gaps likely (20/20 points)")
        elif avg_points > 100:
            quality_score += 15
            print("✓ Reasonable data density (15/20 points)")
        else:
            quality_score += 10
            print("- Sparse data files, gaps possible (10/20 points)")
    
    print(f"\nOVERALL DATA QUALITY SCORE: {quality_score}/100")
    
    if quality_score >= 90:
        print("[EXCELLENT] Premium-grade historical dataset")
    elif quality_score >= 75:
        print("[GOOD] High-quality dataset suitable for professional use")
    elif quality_score >= 60:
        print("[ACCEPTABLE] Adequate dataset with some limitations")
    else:
        print("[NEEDS IMPROVEMENT] Dataset quality below recommended standards")
    
    print(f"\nACCESS RECOMMENDATIONS:")
    print(f"1. Primary data source: {best_bucket}")
    print(f"2. Backup data sources: Other analyzed buckets")
    print(f"3. API endpoint: /api/data/historical")
    print(f"4. Dashboard access: http://localhost:3004")
    
    return 0 if quality_score >= 75 else 1

if __name__ == "__main__":
    exit(main())