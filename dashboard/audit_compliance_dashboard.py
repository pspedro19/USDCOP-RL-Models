#!/usr/bin/env python3
"""
Audit Compliance Monitoring Dashboard
=====================================
Real-time monitoring of L5 pipeline audit requirements and compliance status.
Provides visual feedback on all audit gates and validation metrics.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import requests
from typing import Dict, List, Optional, Any
import logging
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class AuditComplianceMonitor:
    """Monitors and reports on L5 pipeline audit compliance"""
    
    def __init__(self):
        self.compliance_status = {
            'overall_status': 'PENDING',
            'last_update': None,
            'validations': {},
            'metrics': {},
            'alerts': []
        }
        
        # Audit requirements from L5
        self.audit_checks = {
            'key_uniqueness': {
                'name': 'Unicidad de Claves',
                'description': 'Validate (episode_id, t_in_episode) uniqueness',
                'status': 'PENDING',
                'details': {}
            },
            'timestamp_uniqueness': {
                'name': 'Timestamps Únicos',
                'description': 'No duplicate timestamps within episodes',
                'status': 'PENDING',
                'details': {}
            },
            'clip_rate_per_feature': {
                'name': 'Clip-rate por Feature',
                'description': 'Individual clip rates for obs_00 to obs_16',
                'status': 'PENDING',
                'details': {}
            },
            'dtype_enforcement': {
                'name': 'Dtype Enforcement',
                'description': 'All observations as float32',
                'status': 'PENDING',
                'details': {}
            },
            'cost_per_split': {
                'name': 'Costos por Split',
                'description': 'Trade costs and slippage per data split',
                'status': 'PENDING',
                'details': {}
            },
            'non_degenerate_reward': {
                'name': 'Reward No-Degenerado',
                'description': 'Reward signal variance and distribution',
                'status': 'PENDING',
                'details': {}
            },
            'embargo_validation': {
                'name': 'Embargo Half-Open',
                'description': 'No train-test leak with 5-day embargo',
                'status': 'PENDING',
                'details': {}
            },
            'runtime_fingerprint': {
                'name': 'Runtime Fingerprint',
                'description': 'SHA256 of final dataset',
                'status': 'PENDING',
                'details': {}
            },
            'performance_gates': {
                'name': 'Performance Gates',
                'description': 'GO/NO-GO decision criteria',
                'status': 'PENDING',
                'details': {}
            },
            'model_serving': {
                'name': 'Serving Bundle',
                'description': 'ONNX model with smoke test',
                'status': 'PENDING',
                'details': {}
            },
            'inference_latency': {
                'name': 'Latencia Real',
                'description': 'P50/P99 latency under 50ms',
                'status': 'PENDING',
                'details': {}
            }
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_compliance, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_compliance(self):
        """Background thread to monitor compliance status"""
        while True:
            try:
                self.update_compliance_status()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error monitoring compliance: {e}")
                time.sleep(60)
    
    def update_compliance_status(self):
        """Fetch and update compliance status from pipeline"""
        try:
            # Fetch from unified data bus
            response = requests.get('http://localhost:5005/api/audit/compliance', timeout=5)
            if response.status_code == 200:
                data = response.json()
                self._process_compliance_data(data)
            else:
                logger.warning(f"Failed to fetch compliance data: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching compliance data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
    
    def _process_compliance_data(self, data: Dict[str, Any]):
        """Process compliance data from pipeline"""
        
        # Update validation statuses
        for check_id, check_data in data.get('validations', {}).items():
            if check_id in self.audit_checks:
                self.audit_checks[check_id]['status'] = check_data.get('status', 'UNKNOWN')
                self.audit_checks[check_id]['details'] = check_data.get('details', {})
        
        # Calculate overall compliance
        statuses = [check['status'] for check in self.audit_checks.values()]
        if all(s == 'PASS' for s in statuses):
            self.compliance_status['overall_status'] = 'COMPLIANT'
        elif any(s == 'FAIL' for s in statuses):
            self.compliance_status['overall_status'] = 'NON_COMPLIANT'
        elif any(s == 'WARN' for s in statuses):
            self.compliance_status['overall_status'] = 'WARNING'
        else:
            self.compliance_status['overall_status'] = 'IN_PROGRESS'
        
        # Update metrics
        self.compliance_status['metrics'] = data.get('metrics', {})
        
        # Process alerts
        self._process_alerts(data.get('alerts', []))
        
        # Update timestamp
        self.compliance_status['last_update'] = datetime.now().isoformat()
        
        # Emit update via WebSocket
        socketio.emit('compliance_update', self.get_compliance_summary())
    
    def _process_alerts(self, alerts: List[Dict]):
        """Process and prioritize alerts"""
        
        # Add new alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            alert['acknowledged'] = False
            self.compliance_status['alerts'].append(alert)
        
        # Keep only last 100 alerts
        self.compliance_status['alerts'] = self.compliance_status['alerts'][-100:]
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
        self.compliance_status['alerts'].sort(
            key=lambda x: (severity_order.get(x.get('severity', 'INFO'), 5), x['timestamp']),
            reverse=True
        )
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get current compliance summary"""
        
        # Count by status
        status_counts = {'PASS': 0, 'FAIL': 0, 'WARN': 0, 'PENDING': 0}
        for check in self.audit_checks.values():
            status_counts[check['status']] += 0
        
        # Calculate compliance score
        total_checks = len(self.audit_checks)
        passed_checks = status_counts['PASS']
        compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        return {
            'overall_status': self.compliance_status['overall_status'],
            'compliance_score': round(compliance_score, 2),
            'last_update': self.compliance_status['last_update'],
            'status_counts': status_counts,
            'checks': self.audit_checks,
            'metrics': self.compliance_status['metrics'],
            'alerts': self.compliance_status['alerts'][:10],  # Last 10 alerts
            'total_alerts': len(self.compliance_status['alerts'])
        }
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.compliance_status['alerts']:
            if alert.get('id') == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                return True
        return False

# Initialize monitor
monitor = AuditComplianceMonitor()

@app.route('/')
def index():
    """Serve main dashboard page"""
    return render_template('audit_compliance.html')

@app.route('/api/compliance/status')
def get_compliance_status():
    """Get current compliance status"""
    return jsonify(monitor.get_compliance_summary())

@app.route('/api/compliance/checks')
def get_compliance_checks():
    """Get detailed compliance checks"""
    return jsonify(monitor.audit_checks)

@app.route('/api/compliance/metrics')
def get_compliance_metrics():
    """Get compliance metrics"""
    return jsonify(monitor.compliance_status['metrics'])

@app.route('/api/compliance/alerts')
def get_alerts():
    """Get compliance alerts"""
    limit = request.args.get('limit', 50, type=int)
    severity = request.args.get('severity', None)
    
    alerts = monitor.compliance_status['alerts']
    
    if severity:
        alerts = [a for a in alerts if a.get('severity') == severity]
    
    return jsonify(alerts[:limit])

@app.route('/api/compliance/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    if monitor.acknowledge_alert(alert_id):
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Alert not found'}), 404

@app.route('/api/compliance/history')
def get_compliance_history():
    """Get historical compliance data"""
    # This would fetch from a database or file storage
    # For now, return sample data
    return jsonify({
        'history': [
            {
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'compliance_score': 85 + np.random.randint(-5, 10),
                'status': 'COMPLIANT' if np.random.random() > 0.3 else 'WARNING'
            }
            for i in range(24, 0, -1)
        ]
    })

@app.route('/api/compliance/report', methods=['POST'])
def generate_report():
    """Generate compliance report"""
    report_type = request.json.get('type', 'summary')
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'type': report_type,
        'compliance_status': monitor.get_compliance_summary(),
        'recommendations': []
    }
    
    # Add recommendations based on current status
    for check_id, check in monitor.audit_checks.items():
        if check['status'] == 'FAIL':
            report['recommendations'].append({
                'check': check['name'],
                'severity': 'HIGH',
                'action': f"Review and fix {check['description']}"
            })
        elif check['status'] == 'WARN':
            report['recommendations'].append({
                'check': check['name'],
                'severity': 'MEDIUM',
                'action': f"Monitor {check['description']}"
            })
    
    return jsonify(report)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('compliance_update', monitor.get_compliance_summary())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('refresh_compliance')
def handle_refresh():
    """Force refresh compliance status"""
    monitor.update_compliance_status()
    emit('compliance_update', monitor.get_compliance_summary())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    logger.info(f"Starting Audit Compliance Dashboard on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)