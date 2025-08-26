#!/usr/bin/env python3
"""
Motor de Backtesting con Datos Reales y Modelos RL
===================================================
Conecta los datos del pipeline con los modelos entrenados para simulación en tiempo real
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Motor de backtesting con datos reales del pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data = None
        self.models = {}
        self.current_position = None
        self.balance = 100000  # Capital inicial USD
        self.initial_balance = 100000
        self.positions_history = []
        self.performance_history = []
        
        # Cargar datos y modelos
        self.load_pipeline_data()
        self.load_models()
        
    def load_pipeline_data(self):
        """Carga datos del pipeline Diamond (más procesados)"""
        data_path = self.project_root / 'data' / 'processed' / 'diamond' / 'USDCOP_ML_READY_FINAL_COMPLETE.csv'
        
        if data_path.exists():
            self.data = pd.read_csv(data_path)
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data = self.data.sort_values('time')
            
            # Calcular precio base desde retornos
            base_price = 4164.67  # Precio base USDCOP
            self.data['price'] = base_price
            
            # Calcular precios desde retornos acumulados
            returns = self.data['target_return_1'].fillna(0)
            cumulative_returns = (1 + returns).cumprod()
            self.data['price'] = base_price * cumulative_returns
            
            # Agregar columnas comerciales
            self.add_commercial_columns()
            
            logger.info(f"✅ Datos cargados: {len(self.data)} filas")
            logger.info(f"📅 Rango: {self.data['time'].min()} a {self.data['time'].max()}")
            logger.info(f"📊 Columnas: {list(self.data.columns[:10])}...")
        else:
            logger.error(f"❌ No se encontraron datos en {data_path}")
            
    def add_commercial_columns(self):
        """Agrega columnas comerciales para el dashboard"""
        
        # CDT Comparison (12% E.A.)
        cdt_annual_rate = 0.12
        periods_per_year = 252 * 24 * 12  # 5-minute periods
        cdt_period_rate = (1 + cdt_annual_rate) ** (1/periods_per_year) - 1
        
        # Calcular retorno acumulado del CDT
        self.data['cdt_return'] = cdt_period_rate
        self.data['cdt_cumulative'] = (1 + self.data['cdt_return']).cumprod()
        self.data['cdt_value'] = self.initial_balance * self.data['cdt_cumulative']
        
        # Calcular portfolio value con retornos reales
        self.data['portfolio_cumulative'] = (1 + self.data['target_return_1'].fillna(0)).cumprod()
        self.data['portfolio_value'] = self.initial_balance * self.data['portfolio_cumulative']
        
        # Métricas de comparación
        self.data['vs_cdt_ratio'] = self.data['portfolio_value'] / self.data['cdt_value']
        self.data['vs_cdt_abs_usd'] = self.data['portfolio_value'] - self.data['cdt_value']
        self.data['vs_cdt_pct'] = (self.data['vs_cdt_ratio'] - 1) * 100
        
        # Consistency Score (ventana móvil de 30 períodos)
        self.data['is_positive'] = (self.data['target_return_1'] > 0).astype(int)
        self.data['consistency_score'] = self.data['is_positive'].rolling(30, min_periods=1).mean() * 100
        
        # Drawdown
        self.data['peak'] = self.data['portfolio_value'].cummax()
        self.data['drawdown'] = (self.data['portfolio_value'] - self.data['peak']) / self.data['peak'] * 100
        
        logger.info("✅ Columnas comerciales agregadas")
        
    def load_models(self):
        """Carga los modelos RL entrenados"""
        models_path = self.project_root / 'models' / 'trained' / 'complete_5_models'
        
        model_files = {
            'TD3': 'TD3_Complete_final.pkl',
            'DQN': 'DQN_Complete_final.pkl',
            'PPO': 'PPO_Complete_final.pkl',
            'A2C': 'A2C_Complete_final.pkl',
            'SAC': 'SAC_Complete_final.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = models_path / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    logger.info(f"✅ Modelo {name} cargado")
                except Exception as e:
                    logger.error(f"❌ Error cargando {name}: {e}")
                    
    def get_model_prediction(self, model_name, features):
        """Obtiene predicción del modelo"""
        if model_name not in self.models:
            # Predicción aleatoria si no hay modelo
            return np.random.choice([0, 1, 2])  # Buy, Hold, Sell
            
        try:
            model = self.models[model_name]
            
            # Preparar features (excluir columnas no numéricas)
            feature_cols = ['stoch_d_14', 'stoch_d_21', 'rsi_7', 'rsi_14', 'rsi_21', 
                          'rsi_30', 'williams_r_21', 'stoch_k_21', 'price_position_20']
            
            feature_vector = features[feature_cols].values.flatten()
            
            # Hacer predicción
            if hasattr(model, 'predict'):
                action = model.predict(feature_vector.reshape(1, -1))
                if isinstance(action, tuple):
                    action = action[0]
                return int(action) if not isinstance(action, int) else action
            else:
                return 1  # Hold por defecto
                
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return 1  # Hold por defecto
            
    def simulate_trading(self, start_idx=0, end_idx=None, model_name='TD3'):
        """Simula trading con el modelo especificado"""
        
        if end_idx is None:
            end_idx = min(start_idx + 1000, len(self.data))
            
        results = {
            'timestamps': [],
            'prices': [],
            'portfolio_values': [],
            'cdt_values': [],
            'actions': [],
            'returns': [],
            'vs_cdt_ratio': [],
            'consistency_scores': [],
            'drawdowns': []
        }
        
        # Reset estado
        self.balance = self.initial_balance
        position_size = 0
        entry_price = 0
        
        for idx in range(start_idx, end_idx):
            row = self.data.iloc[idx]
            
            # Obtener predicción del modelo
            action = self.get_model_prediction(model_name, row)
            action_name = ['BUY', 'HOLD', 'SELL'][action]
            
            # Ejecutar acción
            current_price = row['price']
            
            if action == 0 and position_size == 0:  # BUY
                position_size = self.balance * 0.95  # Usar 95% del balance
                entry_price = current_price
                self.balance *= 0.05  # Mantener 5% en cash
                
            elif action == 2 and position_size > 0:  # SELL
                # Calcular ganancia/pérdida
                price_change = (current_price - entry_price) / entry_price
                profit = position_size * price_change
                self.balance += position_size + profit
                position_size = 0
                entry_price = 0
                
            # Calcular valor del portfolio
            portfolio_value = self.balance
            if position_size > 0:
                current_value = position_size * (current_price / entry_price)
                portfolio_value += current_value
                
            # Guardar resultados
            results['timestamps'].append(row['time'])
            results['prices'].append(current_price)
            results['portfolio_values'].append(portfolio_value)
            results['cdt_values'].append(row['cdt_value'])
            results['actions'].append(action_name)
            results['returns'].append(row['target_return_1'])
            results['vs_cdt_ratio'].append(portfolio_value / row['cdt_value'])
            results['consistency_scores'].append(row['consistency_score'])
            results['drawdowns'].append(row['drawdown'])
            
        return results
        
    def calculate_metrics(self, results):
        """Calcula métricas de performance"""
        
        portfolio_values = np.array(results['portfolio_values'])
        cdt_values = np.array(results['cdt_values'])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Métricas principales
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
        cdt_return = (cdt_values[-1] - cdt_values[0]) / cdt_values[0] * 100
        
        # Sharpe Ratio (anualizado)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
            
        # Máximo Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Consistency
        positive_periods = sum(1 for r in returns if r > 0)
        consistency = positive_periods / len(returns) * 100 if len(returns) > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'cdt_return': cdt_return,
            'vs_cdt_ratio': portfolio_values[-1] / cdt_values[-1],
            'vs_cdt_abs_usd': portfolio_values[-1] - cdt_values[-1],
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'consistency_score': consistency,
            'final_value': portfolio_values[-1],
            'cdt_final_value': cdt_values[-1]
        }
        
        return metrics

# API para el dashboard
def get_backtest_data(periods=500, model='TD3'):
    """API para obtener datos de backtest"""
    engine = BacktestEngine()
    
    # Simular trading
    results = engine.simulate_trading(
        start_idx=len(engine.data) - periods,
        end_idx=len(engine.data),
        model_name=model
    )
    
    # Calcular métricas
    metrics = engine.calculate_metrics(results)
    
    # Formatear para el dashboard - convertir numpy types a Python natives
    response = {
        'success': True,
        'data': {
            'timestamps': [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in results['timestamps'][-100:]],  # Últimos 100
            'prices': [float(p) for p in results['prices'][-100:]],
            'portfolio_values': [float(v) for v in results['portfolio_values'][-100:]],
            'cdt_values': [float(v) for v in results['cdt_values'][-100:]],
            'actions': results['actions'][-100:],
            'vs_cdt_ratio': [float(r) for r in results['vs_cdt_ratio'][-100:]],
            'consistency_scores': [float(s) for s in results['consistency_scores'][-100:]]
        },
        'metrics': {
            'total_return': float(metrics['total_return']),
            'cdt_return': float(metrics['cdt_return']),
            'vs_cdt_ratio': float(metrics['vs_cdt_ratio']),
            'vs_cdt_abs_usd': float(metrics['vs_cdt_abs_usd']),
            'sharpe_ratio': float(metrics['sharpe_ratio']),
            'max_drawdown': float(metrics['max_drawdown']),
            'consistency_score': float(metrics['consistency_score']),
            'final_value': float(metrics['final_value']),
            'cdt_final_value': float(metrics['cdt_final_value'])
        },
        'summary': {
            'model': model,
            'periods': periods,
            'initial_capital': engine.initial_balance,
            'final_value': f"${metrics['final_value']:,.2f}",
            'total_return': f"{metrics['total_return']:.2f}%",
            'vs_cdt': f"{metrics['vs_cdt_ratio']:.2f}x",
            'sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'max_dd': f"{metrics['max_drawdown']:.2f}%"
        }
    }
    
    return response

if __name__ == "__main__":
    # Test del motor
    engine = BacktestEngine()
    
    print("\n🚀 BACKTESTING ENGINE INICIADO")
    print("=" * 50)
    
    # Probar simulación
    results = engine.simulate_trading(
        start_idx=len(engine.data) - 1000,
        end_idx=len(engine.data),
        model_name='TD3'
    )
    
    # Calcular métricas
    metrics = engine.calculate_metrics(results)
    
    print(f"\n📊 RESULTADOS DEL BACKTEST:")
    print(f"Capital Inicial: ${engine.initial_balance:,.2f}")
    print(f"Valor Final: ${metrics['final_value']:,.2f}")
    print(f"Retorno Total: {metrics['total_return']:.2f}%")
    print(f"vs CDT 12%: {metrics['vs_cdt_ratio']:.2f}x")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Consistencia: {metrics['consistency_score']:.1f}%")