#!/usr/bin/env python3
"""
USDCOP Trading System - Main Dashboard Server
=============================================
Professional trading dashboard server with clean architecture
"""

import socket
import threading
import time
import webbrowser
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class USDCOPDashboardServer:
    def __init__(self):
        self.app = Flask(__name__, static_folder=None)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        self.port = None
        self.server_thread = None
        self.is_running = False
        
        # Setup routes
        self.setup_routes()
        
    def find_available_port(self, start_port=8000, max_attempts=100):
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    logger.info(f"Found available port: {port}")
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Serve the main stable dashboard"""
            try:
                dashboard_path = Path(__file__).parent / "stable" / "main_dashboard.html"
                
                if dashboard_path.exists():
                    with open(dashboard_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Fix WebSocket connection to use current port
                    content = content.replace('ws://localhost:5002/ws', f'ws://localhost:{self.port}/ws')
                    content = content.replace('ws://localhost:8082/ws', f'ws://localhost:{self.port}/ws')
                    
                    logger.info(f"Serving dashboard: {dashboard_path.name}")
                    return content
                else:
                    logger.error(f"Dashboard file not found: {dashboard_path}")
                    return self.get_fallback_dashboard()
            except Exception as e:
                logger.error(f"Error loading dashboard: {e}")
                return self.get_fallback_dashboard()
        
        @self.app.route('/api/status')
        def get_status():
            """Get server status"""
            return jsonify({
                'status': 'running',
                'port': self.port,
                'dashboard': 'main',
                'websocket': True,
                'version': '1.0.0'
            })
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({'status': 'healthy', 'timestamp': time.time()})
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get current performance metrics"""
            return jsonify({
                'current_price': 4164.67,
                'total_return': '30.8%',
                'vs_cdt': '2.57x',
                'consistency': '95%',
                'max_drawdown': '-8.5%',
                'recovery_time': '2.3 months'
            })
        
        @self.app.route('/<path:filename>')
        def serve_file(filename):
            """Serve specific files if requested"""
            if filename.endswith('.html') or filename.endswith('.js') or filename.endswith('.css'):
                try:
                    file_path = Path(__file__).parent / filename
                    if file_path.exists():
                        return send_from_directory(Path(__file__).parent, filename)
                except Exception as e:
                    logger.error(f"Error serving file {filename}: {e}")
            
            # For any other file, redirect to main dashboard
            return self.app.redirect('/')
    
    def get_fallback_dashboard(self):
        """Return a simple fallback dashboard if the main one fails"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>USDCOP Trading Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; background: #000; color: #fff; padding: 50px; }
                .container { max-width: 800px; margin: 0 auto; text-align: center; }
                .error { background: #333; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .success { background: #0a0; padding: 20px; border-radius: 10px; margin: 20px 0; }
                button { background: #00d9ff; color: #000; border: none; padding: 15px 30px; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }
                button:hover { background: #00b8e6; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 USDCOP Trading Dashboard</h1>
                <div class="success">
                    <h2>✅ Server Running Successfully!</h2>
                    <p>Port: <strong>{port}</strong></p>
                    <p>Status: <strong>Active</strong></p>
                </div>
                <div class="error">
                    <h3>⚠️ Main Dashboard Not Available</h3>
                    <p>The main dashboard file could not be loaded, but the server is working.</p>
                </div>
                <button onclick="window.location.reload()">🔄 Refresh Page</button>
                <button onclick="window.location.href='/api/status'">📊 Check Status</button>
            </div>
        </body>
        </html>
        """.format(port=self.port)
    
    def start_server(self, port=None):
        """Start the dashboard server"""
        try:
            if port is None:
                self.port = self.find_available_port(8000)
            else:
                self.port = port
            
            logger.info(f"Starting USDCOP Dashboard Server on port {self.port}")
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Open browser
            url = f"http://localhost:{self.port}"
            logger.info(f"Dashboard available at: {url}")
            
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                print(f"\n🌐 Open your browser and go to: {url}")
            
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def _run_server(self):
        """Internal method to run the Flask server"""
        try:
            self.socketio.run(
                self.app,
                host='0.0.0.0',
                port=self.port,
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"Server error: {e}")
            self.is_running = False
    
    def stop_server(self):
        """Stop the dashboard server"""
        self.is_running = False
        logger.info("Stopping USDCOP Dashboard Server")
    
    def get_status(self):
        """Get current server status"""
        return {
            'is_running': self.is_running,
            'port': self.port,
            'url': f"http://localhost:{self.port}" if self.port else None
        }

def main():
    """Main function"""
    print("🚀 USDCOP Trading System - Main Dashboard Server")
    print("=" * 60)
    
    # Create and start server
    server = USDCOPDashboardServer()
    
    if server.start_server():
        print(f"✅ Dashboard server started successfully!")
        print(f"🌐 URL: http://localhost:{server.port}")
        print(f"💼 Dashboard: Professional Trading Interface")
        print("\n📱 Features:")
        print("   • Professional Trading UI")
        print("   • Real-time WebSocket support")
        print("   • Dynamic port allocation")
        print("   • Clean, professional architecture")
        print("\n⏹️  Press Ctrl+C to stop the server")
        
        try:
            # Keep the main thread alive
            while server.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping server...")
            server.stop_server()
            print("✅ Server stopped")
    
    else:
        print("❌ Failed to start dashboard server")
        sys.exit(1)

if __name__ == '__main__':
    main()
