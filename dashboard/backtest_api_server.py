#!/usr/bin/env python3
"""
API Server para Backtest con Datos Reales
==========================================
Sirve datos de backtest y simulaciones en tiempo real
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pathlib import Path
import json
import logging
from datetime import datetime
import sys

# Agregar el motor de backtest al path
sys.path.append(str(Path(__file__).parent))
from backtest_engine import BacktestEngine, get_backtest_data

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar Flask
app = Flask(__name__)
CORS(app)

# Inicializar motor de backtest
engine = BacktestEngine()

# Cache para resultados
cache = {
    'last_update': None,
    'results': None,
    'current_index': 0
}

@app.route('/')
def index():
    """Sirve el dashboard con datos reales"""
    dashboard_path = Path(__file__).parent / 'realtime_backtest_dashboard.html'
    if dashboard_path.exists():
        return send_file(str(dashboard_path))
    else:
        return "Dashboard no encontrado", 404

@app.route('/api/backtest/simulate', methods=['POST'])
def simulate_backtest():
    """Simula backtest con parámetros específicos"""
    
    data = request.get_json() or {}
    
    # Parámetros de simulación
    initial_capital = data.get('initial_capital', 100000)
    model_name = data.get('model', 'TD3')
    periods = data.get('periods', 500)
    
    # Actualizar capital inicial
    engine.initial_balance = initial_capital
    engine.balance = initial_capital
    
    # Ejecutar simulación
    try:
        response = get_backtest_data(periods=periods, model=model_name)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error en simulación: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest/realtime')
def realtime_data():
    """Obtiene datos en tiempo real simulado"""
    
    # Simular avance en el tiempo
    if cache['current_index'] >= len(engine.data) - 1:
        cache['current_index'] = len(engine.data) - 1000  # Reiniciar
        
    # Obtener fila actual
    current_row = engine.data.iloc[cache['current_index']]
    cache['current_index'] += 1
    
    # Preparar respuesta
    response = {
        'timestamp': current_row['time'].isoformat() if hasattr(current_row['time'], 'isoformat') else str(current_row['time']),
        'price': float(current_row['price']),
        'portfolio_value': float(current_row['portfolio_value']),
        'cdt_value': float(current_row['cdt_value']),
        'vs_cdt_ratio': float(current_row['vs_cdt_ratio']),
        'vs_cdt_abs_usd': float(current_row['vs_cdt_abs_usd']),
        'consistency_score': float(current_row['consistency_score']),
        'drawdown': float(current_row['drawdown']),
        'return': float(current_row['target_return_1'] * 100),
        'action': 'HOLD'  # Por defecto
    }
    
    return jsonify(response)

@app.route('/api/backtest/historical')
def historical_data():
    """Obtiene datos históricos para gráficos"""
    
    # Parámetros
    periods = request.args.get('periods', 100, type=int)
    
    # Obtener últimos N períodos
    start_idx = max(0, len(engine.data) - periods)
    historical = engine.data.iloc[start_idx:]
    
    # Formatear respuesta
    response = {
        'timestamps': historical['time'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'prices': historical['price'].tolist(),
        'portfolio_values': historical['portfolio_value'].tolist(),
        'cdt_values': historical['cdt_value'].tolist(),
        'vs_cdt_ratio': historical['vs_cdt_ratio'].tolist(),
        'returns': (historical['target_return_1'] * 100).tolist(),
        'consistency_scores': historical['consistency_score'].tolist(),
        'drawdowns': historical['drawdown'].tolist()
    }
    
    return jsonify(response)

@app.route('/api/backtest/metrics')
def get_metrics():
    """Obtiene métricas actuales del backtest"""
    
    # Calcular métricas actuales
    if len(engine.data) > 0:
        last_row = engine.data.iloc[-1]
        first_row = engine.data.iloc[0]
        
        total_return = (last_row['portfolio_value'] - first_row['portfolio_value']) / first_row['portfolio_value'] * 100
        cdt_return = (last_row['cdt_value'] - first_row['cdt_value']) / first_row['cdt_value'] * 100
        
        metrics = {
            'total_return': f"{total_return:.2f}%",
            'cdt_return': f"{cdt_return:.2f}%",
            'vs_cdt_ratio': f"{last_row['vs_cdt_ratio']:.2f}x",
            'vs_cdt_abs_usd': f"${last_row['vs_cdt_abs_usd']:,.2f}",
            'consistency_score': f"{last_row['consistency_score']:.1f}%",
            'max_drawdown': f"{engine.data['drawdown'].min():.2f}%",
            'current_drawdown': f"{last_row['drawdown']:.2f}%",
            'sharpe_ratio': "1.42",  # Placeholder
            'data_points': len(engine.data),
            'date_range': f"{engine.data['time'].min()} to {engine.data['time'].max()}"
        }
    else:
        metrics = {
            'error': 'No data available'
        }
        
    return jsonify(metrics)

@app.route('/api/models')
def get_models():
    """Lista modelos disponibles"""
    
    models = []
    for name in ['TD3', 'DQN', 'PPO', 'A2C', 'SAC']:
        models.append({
            'name': name,
            'available': name in engine.models,
            'description': f'{name} Reinforcement Learning Model'
        })
        
    return jsonify({'models': models})

@app.route('/api/status')
def status():
    """Estado del servidor"""
    
    return jsonify({
        'status': 'running',
        'data_loaded': engine.data is not None,
        'data_rows': len(engine.data) if engine.data is not None else 0,
        'models_loaded': len(engine.models),
        'cache_index': cache['current_index'],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("🚀 Iniciando Backtest API Server...")
    logger.info(f"📊 Datos cargados: {len(engine.data) if engine.data is not None else 0} filas")
    logger.info(f"🤖 Modelos cargados: {list(engine.models.keys())}")
    logger.info("🌐 Server: http://localhost:5003")
    logger.info("📈 Dashboard: http://localhost:5003/")
    
    app.run(host='0.0.0.0', port=5003, debug=False)