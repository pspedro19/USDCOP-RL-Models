#!/usr/bin/env python3
"""
Pipeline Health Monitoring Service Launcher
===========================================

Launches the Pipeline Health Monitoring API service with proper configuration
and environment setup for production or development modes.
"""

import os
import sys
import subprocess
import signal
import time
import logging
from pathlib import Path
import yaml
import argparse

# Add services directory to Python path
services_dir = Path(__file__).parent.parent / "services"
sys.path.insert(0, str(services_dir))

def setup_logging(log_level='INFO'):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(__file__).parent.parent / "logs" / "pipeline_health_service.log")
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'psycopg2-binary',
        'redis',
        'boto3',
        'aiohttp',
        'pydantic',
        'python-multipart'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_environment():
    """Check if required environment variables and services are available"""
    config_file = Path(__file__).parent.parent / "config" / "pipeline_health_config.yaml"
    
    if not config_file.exists():
        print(f"Configuration file not found: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading configuration: {e}")
        return False
    
    # Check database connection
    try:
        import psycopg2
        db_config = config['database']
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            connect_timeout=5
        )
        conn.close()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
    
    # Check Redis connection
    try:
        import redis
        redis_config = config['redis']
        r = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            password=redis_config.get('password'),
            socket_timeout=5
        )
        r.ping()
        print("✓ Redis connection successful")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False
    
    # Check S3/MinIO connection
    try:
        import boto3
        from botocore.client import Config
        s3_config = config['s3']
        client = boto3.client(
            's3',
            endpoint_url=s3_config['endpoint'],
            aws_access_key_id=s3_config['access_key'],
            aws_secret_access_key=s3_config['secret_key'],
            config=Config(signature_version='s3v4')
        )
        client.list_buckets()
        print("✓ S3/MinIO connection successful")
    except Exception as e:
        print(f"✗ S3/MinIO connection failed: {e}")
        return False
    
    return True

def run_service(host='0.0.0.0', port=8002, log_level='info', reload=False):
    """Run the Pipeline Health API service"""
    
    # Set up environment
    os.environ['PYTHONPATH'] = str(services_dir)
    
    # Build uvicorn command
    cmd = [
        sys.executable, '-m', 'uvicorn',
        'pipeline_health_api:app',
        '--host', host,
        '--port', str(port),
        '--log-level', log_level,
    ]
    
    if reload:
        cmd.append('--reload')
    
    print(f"Starting Pipeline Health API service on {host}:{port}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Change to services directory
        os.chdir(services_dir)
        
        # Start the service
        process = subprocess.Popen(cmd)
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print("\nShutting down Pipeline Health API service...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for the process
        process.wait()
        
    except Exception as e:
        print(f"Error starting service: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Pipeline Health Monitoring Service Launcher')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8002, help='Port to bind to (default: 8002)')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'],
                       help='Log level (default: info)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies and environment')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and environment checks')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level.upper())
    logger = logging.getLogger(__name__)
    
    print("Pipeline Health Monitoring Service Launcher")
    print("=" * 50)
    
    if not args.skip_checks:
        print("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("Checking environment...")
        if not check_environment():
            sys.exit(1)
    
    if args.check_only:
        print("✓ All checks passed! Service is ready to run.")
        return
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    print("All checks passed! Starting service...")
    run_service(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload
    )

if __name__ == "__main__":
    main()