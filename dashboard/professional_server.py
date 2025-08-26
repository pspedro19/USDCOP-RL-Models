#!/usr/bin/env python3
"""
Professional Trading Dashboard Server
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, send_file, jsonify, request
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Dashboard path
DASHBOARD_FILE = Path(__file__).parent / 'dashboard-trading-profesional.html'

@app.route('/')
def index():
    """Serve the professional trading dashboard"""
    if DASHBOARD_FILE.exists():
        logger.info(f"Serving dashboard from: {DASHBOARD_FILE}")
        return send_file(str(DASHBOARD_FILE))
    else:
        logger.error(f"Dashboard not found at: {DASHBOARD_FILE}")
        return f"Dashboard not found at {DASHBOARD_FILE}", 404

@app.route('/api/status')
def status():
    """API endpoint for system status"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'server': 'professional_server'
    })

@app.route('/api/institutional-metrics')
def institutional_metrics():
    """Mock institutional metrics for dashboard"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'sharpe_ratio': 1.85,
            'sortino_ratio': 2.13,
            'calmar_ratio': 2.45,
            'max_drawdown': -3.2,
            'var_95': -1.5,
            'volatility': 12.8
        }
    })

@app.route('/api/current-data')
def current_data():
    """Mock current trading data"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'price': 4164.67,
        'returns': 0.23,
        'balance': 10325.40,
        'action': 'HOLD',
        'confidence': 0.78
    })

if __name__ == '__main__':
    logger.info(f"🚀 Starting Professional Trading Dashboard Server...")
    logger.info(f"📁 Dashboard file: {DASHBOARD_FILE}")
    logger.info(f"✅ File exists: {DASHBOARD_FILE.exists()}")
    logger.info(f"🌐 Dashboard: http://localhost:5002")
    
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")