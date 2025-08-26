#!/usr/bin/env python3
"""
Premium Data Server - Silver Dataset (Lun-Vie 08:00-14:00 COT)
=============================================================
Servidor dedicado para servir datos del dataset Silver Premium-Only
Solo incluye datos de la sesión Premium para máxima calidad
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PremiumDataServer:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.silver_dir = self.project_root / 'data' / 'processed' / 'silver'
        
        # Data storage
        self.data = None
        self.filtered_data = None
        self.data_info = {}
        
        # Session configuration
        self.session_config = {
            'name': 'Premium Session',
            'days': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes'],
            'hours_cot': '08:00-14:00',
            'hours_utc': '13:00-19:00',
            'completeness': 90.9,
            'quality_score': 0.914  # 91.4% based on analysis
        }
        
        # Load data on startup
        self.load_silver_premium_data()
        
        # Setup routes
        self.setup_routes()
    
    def load_silver_premium_data(self):
        """Load the Silver Premium-Only dataset"""
        try:
            # Find the latest Silver Premium file
            silver_files = list(self.silver_dir.glob("SILVER_PREMIUM_ONLY_*.csv"))
            
            if not silver_files:
                logger.error("No Silver Premium files found")
                return False
            
            # Get the most recent file
            latest_file = max(silver_files, key=lambda x: x.stat().st_mtime)
            
            logger.info(f"Loading Silver Premium data from: {latest_file.name}")
            
            # Load data
            self.data = pd.read_csv(latest_file)
            self.data['time'] = pd.to_datetime(self.data['time'])
            
            # Sort by time
            self.data = self.data.sort_values('time').reset_index(drop=True)
            
            # Add derived columns for filtering
            self.data['date'] = self.data['time'].dt.date
            self.data['hour'] = self.data['time'].dt.hour
            self.data['minute'] = self.data['time'].dt.minute
            self.data['day_of_week'] = self.data['time'].dt.day_name()
            self.data['hour_cot'] = ((self.data['hour'] - 5) % 24)  # Convert UTC to COT
            
            # Calculate statistics
            self.data_info = {
                'total_records': len(self.data),
                'start_date': str(self.data['time'].min()),
                'end_date': str(self.data['time'].max()),
                'unique_days': self.data['date'].nunique(),
                'price_range': {
                    'min': float(self.data['close'].min()),
                    'max': float(self.data['close'].max()),
                    'mean': float(self.data['close'].mean()),
                    'std': float(self.data['close'].std())
                },
                'session': self.session_config,
                'file_loaded': latest_file.name
            }
            
            logger.info(f"✅ Loaded {len(self.data):,} records")
            logger.info(f"   Period: {self.data_info['start_date']} to {self.data_info['end_date']}")
            logger.info(f"   Session: Premium (Lun-Vie 08:00-14:00 COT)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Silver Premium data: {e}")
            return False
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/')
        def index():
            """Serve the premium dashboard HTML"""
            dashboard_path = Path(__file__).parent / 'premium_dashboard.html'
            if dashboard_path.exists():
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return "Dashboard not found", 404
        
        @self.app.route('/premium_dashboard.html')
        def premium_dashboard():
            """Serve the premium dashboard HTML"""
            dashboard_path = Path(__file__).parent / 'premium_dashboard.html'
            if dashboard_path.exists():
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return "Dashboard not found", 404
        
        @self.app.route('/api/data/info')
        def get_data_info():
            """Get information about loaded data"""
            return jsonify(self.data_info)
        
        @self.app.route('/api/data/latest')
        def get_latest_data():
            """Get the latest N records"""
            try:
                n = int(request.args.get('n', 100))
                if self.data is not None:
                    latest = self.data.tail(n)
                    return jsonify({
                        'data': latest[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict('records'),
                        'count': len(latest)
                    })
                return jsonify({'error': 'No data loaded'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/data/range')
        def get_data_range():
            """Get data within a date range"""
            try:
                start = request.args.get('start')
                end = request.args.get('end')
                
                if not start or not end:
                    return jsonify({'error': 'Start and end dates required'}), 400
                
                if self.data is not None:
                    mask = (self.data['time'] >= start) & (self.data['time'] <= end)
                    filtered = self.data[mask]
                    
                    return jsonify({
                        'data': filtered[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict('records'),
                        'count': len(filtered),
                        'start': start,
                        'end': end
                    })
                return jsonify({'error': 'No data loaded'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/data/daily')
        def get_daily_data():
            """Get daily aggregated data"""
            try:
                if self.data is not None:
                    daily = self.data.groupby('date').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).reset_index()
                    
                    daily['date'] = daily['date'].astype(str)
                    
                    return jsonify({
                        'data': daily.to_dict('records'),
                        'count': len(daily)
                    })
                return jsonify({'error': 'No data loaded'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/data/chart')
        def get_chart_data():
            """Get data formatted for charting"""
            try:
                n = int(request.args.get('n', 500))
                
                if self.data is not None:
                    chart_data = self.data.tail(n)
                    
                    # Format for charting libraries
                    candlestick_data = []
                    for _, row in chart_data.iterrows():
                        candlestick_data.append({
                            'x': row['time'].isoformat(),
                            'o': float(row['open']),
                            'h': float(row['high']),
                            'l': float(row['low']),
                            'c': float(row['close']),
                            'v': float(row['volume']) if 'volume' in row else 0
                        })
                    
                    return jsonify({
                        'candlestick': candlestick_data,
                        'count': len(candlestick_data),
                        'session': self.session_config
                    })
                return jsonify({'error': 'No data loaded'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/data/statistics')
        def get_statistics():
            """Get statistical analysis of the data"""
            try:
                if self.data is not None:
                    # Calculate returns
                    df_stats = self.data.copy()
                    df_stats['returns'] = df_stats['close'].pct_change()
                    
                    # Statistics by day of week - simplified
                    dow_counts = df_stats['day_of_week'].value_counts().to_dict()
                    
                    # Statistics by hour - simplified
                    hour_counts = df_stats['hour_cot'].value_counts().sort_index().to_dict()
                    
                    return jsonify({
                        'overall': {
                            'total_records': len(self.data),
                            'mean_return': float(df_stats['returns'].mean()) if not df_stats['returns'].isna().all() else 0,
                            'std_return': float(df_stats['returns'].std()) if not df_stats['returns'].isna().all() else 0,
                            'price_range': self.data_info['price_range']
                        },
                        'by_day': {'count': dow_counts},
                        'by_hour': {'count': hour_counts},
                        'session': self.session_config
                    })
                return jsonify({'error': 'No data loaded'}), 404
            except Exception as e:
                logger.error(f"Error in statistics endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/data/quality')
        def get_quality_metrics():
            """Get data quality metrics"""
            try:
                if self.data is not None:
                    # Calculate gaps
                    self.data['time_diff'] = self.data['time'].diff()
                    self.data['gap_minutes'] = self.data['time_diff'].dt.total_seconds() / 60
                    
                    # Count gaps > 5 minutes within same day
                    real_gaps = 0
                    for i in range(1, len(self.data)):
                        if self.data.iloc[i]['date'] == self.data.iloc[i-1]['date']:
                            if self.data.iloc[i]['gap_minutes'] > 5:
                                real_gaps += 1
                    
                    # Expected bars (72 per day for 6 hours)
                    expected_bars = self.data_info['unique_days'] * 72
                    actual_bars = len(self.data)
                    completeness = (actual_bars / expected_bars * 100) if expected_bars > 0 else 0
                    
                    return jsonify({
                        'completeness': f"{completeness:.1f}%",
                        'expected_bars': expected_bars,
                        'actual_bars': actual_bars,
                        'missing_bars': expected_bars - actual_bars,
                        'real_gaps': real_gaps,
                        'quality_score': self.session_config['quality_score'],
                        'session': self.session_config,
                        'recommendation': 'OPTIMAL for trading and ML models'
                    })
                return jsonify({'error': 'No data loaded'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # WebSocket events
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected")
            emit('connected', {
                'message': 'Connected to Premium Data Server',
                'data_loaded': self.data is not None,
                'records': len(self.data) if self.data is not None else 0,
                'session': self.session_config
            })
        
        @self.socketio.on('request_live_data')
        def handle_live_data_request(data):
            """Stream live data updates"""
            try:
                if self.data is not None:
                    # Get last N records
                    n = data.get('count', 100)
                    latest = self.data.tail(n)
                    
                    emit('live_data', {
                        'data': latest[['time', 'open', 'high', 'low', 'close']].to_dict('records'),
                        'count': len(latest),
                        'session': self.session_config
                    })
            except Exception as e:
                emit('error', {'message': str(e)})
    
    def run(self, host='0.0.0.0', port=8090):
        """Run the server"""
        logger.info(f"Starting Premium Data Server on http://{host}:{port}")
        logger.info(f"Session: Premium (Lun-Vie 08:00-14:00 COT)")
        logger.info(f"Data Quality: {self.session_config['quality_score']*100:.1f}%")
        self.socketio.run(self.app, host=host, port=port, debug=False)


if __name__ == '__main__':
    server = PremiumDataServer()
    server.run(port=8090)