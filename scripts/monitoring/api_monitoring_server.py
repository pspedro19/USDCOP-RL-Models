"""
API Monitoring Server
====================
Flask server to provide API monitoring data to the frontend dashboard.
This server bridges the Python backend monitoring with the React frontend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import sys
import os
from datetime import datetime, timedelta
import json

# Add the airflow dags path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'airflow', 'dags'))

try:
    from utils.enhanced_api_monitor import api_monitor
    from data_sources.twelvedata_client import TwelveDataClient
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    print(f"Warning: Enhanced monitoring not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'monitoring_available': MONITORING_AVAILABLE
    })

# API monitoring endpoints
@app.route('/api/monitoring/status', methods=['GET'])
def get_monitoring_status():
    """Get comprehensive API monitoring status"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        # Get all API key statuses
        key_statuses = api_monitor.get_all_key_statuses()
        
        # Get health metrics for each API
        api_names = set(status.api_name for status in key_statuses)
        health_metrics = {}
        for api_name in api_names:
            health_metrics[api_name] = api_monitor.get_api_health_metrics(api_name).__dict__
        
        # Convert dataclass objects to dictionaries
        key_statuses_dict = [status.__dict__ for status in key_statuses]
        
        # Convert datetime objects to ISO format
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        for status in key_statuses_dict:
            for key, value in status.items():
                status[key] = convert_datetime(value)
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'key_statuses': key_statuses_dict,
            'health_metrics': health_metrics,
            'summary': {
                'total_keys': len(key_statuses),
                'active_keys': len([s for s in key_statuses if s.status == 'ACTIVE']),
                'rate_limited_keys': len([s for s in key_statuses if s.status == 'RATE_LIMITED']),
                'error_keys': len([s for s in key_statuses if s.status == 'ERROR']),
                'warning_keys': len([s for s in key_statuses if s.status == 'WARNING'])
            }
        })
    
    except Exception as e:
        logging.error(f"Error getting monitoring status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/health/<api_name>', methods=['GET'])
def get_api_health(api_name):
    """Get health metrics for a specific API"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        health_metrics = api_monitor.get_api_health_metrics(api_name)
        return jsonify(health_metrics.__dict__)
    
    except Exception as e:
        logging.error(f"Error getting API health for {api_name}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/keys', methods=['GET'])
def get_key_statuses():
    """Get all API key statuses"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        key_statuses = api_monitor.get_all_key_statuses()
        
        # Convert to dictionaries and handle datetime objects
        statuses_dict = []
        for status in key_statuses:
            status_dict = status.__dict__.copy()
            if status_dict.get('last_used') and isinstance(status_dict['last_used'], datetime):
                status_dict['last_used'] = status_dict['last_used'].isoformat()
            statuses_dict.append(status_dict)
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'key_statuses': statuses_dict
        })
    
    except Exception as e:
        logging.error(f"Error getting key statuses: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/key/<key_id>', methods=['GET'])
def get_key_status(key_id):
    """Get status of a specific API key"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        key_status = api_monitor.get_key_status(key_id)
        
        if key_status:
            status_dict = key_status.__dict__.copy()
            if status_dict.get('last_used') and isinstance(status_dict['last_used'], datetime):
                status_dict['last_used'] = status_dict['last_used'].isoformat()
            return jsonify(status_dict)
        else:
            return jsonify({'error': f'Key {key_id} not found'}), 404
    
    except Exception as e:
        logging.error(f"Error getting key status for {key_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/best-key/<api_name>', methods=['GET'])
def get_best_key(api_name):
    """Get the best available API key for an API"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        best_key = api_monitor.get_best_available_key(api_name)
        
        return jsonify({
            'api_name': api_name,
            'best_key': best_key,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error getting best key for {api_name}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/export', methods=['GET'])
def export_metrics():
    """Export all monitoring metrics as JSON"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        metrics_json = api_monitor.export_metrics_json()
        return jsonify(json.loads(metrics_json))
    
    except Exception as e:
        logging.error(f"Error exporting metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/alerts', methods=['GET'])
def get_alerts():
    """Get current alerts and warnings"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        key_statuses = api_monitor.get_all_key_statuses()
        alerts = []
        
        for status in key_statuses:
            # Rate limit warnings
            daily_usage_percent = status.daily_calls / status.daily_limit if status.daily_limit > 0 else 0
            if daily_usage_percent > 0.85:
                alerts.append({
                    'type': 'RATE_LIMIT_WARNING',
                    'severity': 'CRITICAL' if daily_usage_percent > 0.95 else 'WARNING',
                    'key_id': status.key_id,
                    'api_name': status.api_name,
                    'message': f'Key {status.key_id} is at {daily_usage_percent:.1%} of daily rate limit',
                    'usage_percent': daily_usage_percent * 100,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Cost warnings
            if status.daily_cost > 100:
                alerts.append({
                    'type': 'COST_WARNING',
                    'severity': 'WARNING',
                    'key_id': status.key_id,
                    'api_name': status.api_name,
                    'message': f'Key {status.key_id} daily cost ${status.daily_cost:.2f} exceeds $100',
                    'daily_cost': status.daily_cost,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Error rate warnings
            if status.success_rate < 90:
                alerts.append({
                    'type': 'ERROR_RATE_WARNING',
                    'severity': 'WARNING',
                    'key_id': status.key_id,
                    'api_name': status.api_name,
                    'message': f'Key {status.key_id} success rate {status.success_rate:.1f}% is below 90%',
                    'success_rate': status.success_rate,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Response time warnings
            if status.avg_response_time > 5000:
                alerts.append({
                    'type': 'RESPONSE_TIME_WARNING',
                    'severity': 'WARNING',
                    'key_id': status.key_id,
                    'api_name': status.api_name,
                    'message': f'Key {status.key_id} avg response time {status.avg_response_time:.0f}ms exceeds 5s',
                    'avg_response_time': status.avg_response_time,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'alert_count': len(alerts),
            'alerts': alerts
        })
    
    except Exception as e:
        logging.error(f"Error getting alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/simulate-call', methods=['POST'])
def simulate_api_call():
    """Simulate an API call for testing purposes"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        data = request.get_json()
        
        api_monitor.record_api_call(
            api_name=data.get('api_name', 'twelvedata'),
            endpoint=data.get('endpoint', 'test'),
            key_id=data.get('key_id', 'key_test'),
            success=data.get('success', True),
            response_time_ms=data.get('response_time_ms', 1000),
            status_code=data.get('status_code', 200),
            error_message=data.get('error_message')
        )
        
        return jsonify({
            'status': 'success',
            'message': 'API call recorded',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error simulating API call: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/reset-daily', methods=['POST'])
def reset_daily_counters():
    """Reset daily counters for all keys (admin function)"""
    if not MONITORING_AVAILABLE:
        return jsonify({'error': 'Enhanced monitoring not available'}), 503
    
    try:
        # This would typically require authentication
        # For now, just log the request
        logging.warning("Daily counter reset requested")
        
        return jsonify({
            'status': 'success',
            'message': 'Daily counters reset request logged',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error resetting daily counters: {e}")
        return jsonify({'error': str(e)}), 500

# Test endpoint for TwelveData integration
@app.route('/api/test/twelvedata', methods=['GET'])
def test_twelvedata():
    """Test TwelveData API with monitoring"""
    try:
        client = TwelveDataClient()
        
        # Test latest price
        price_data = client.get_latest_price()
        
        # Test monitoring status
        monitoring_status = client.get_monitoring_status()
        
        return jsonify({
            'status': 'success',
            'price_data': price_data,
            'monitoring_status': monitoring_status,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error testing TwelveData: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting API Monitoring Server...")
    logging.info(f"Monitoring Available: {MONITORING_AVAILABLE}")
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=8001,
        debug=True
    )