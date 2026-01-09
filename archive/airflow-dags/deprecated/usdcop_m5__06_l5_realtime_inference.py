"""
DAG: usdcop_m5__06_l5_realtime_inference
================================================================================
Ejecuta inferencia del modelo RL PPO en tiempo real cada 5 minutos durante
horario de mercado colombiano (8:00 AM - 12:55 PM COT, Lunes a Viernes).

Flujo:
1. Verifica si es horario de mercado
2. Obtiene últimos datos OHLCV + features
3. Normaliza observación (20 features)
4. Ejecuta modelo PPO
5. Almacena inferencia y acción en DB
6. Actualiza curva de equity
7. Genera alertas si es necesario

Tablas actualizadas:
- dw.fact_rl_inference
- dw.fact_agent_actions
- dw.fact_equity_curve_realtime
- dw.fact_session_performance (agregado)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import numpy as np
import json
import sys
import os

# Agregar path para imports locales
sys.path.insert(0, os.path.dirname(__file__))

DAG_ID = 'usdcop_m5__06_l5_realtime_inference'

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    'model_path': '/opt/airflow/models/ppo_usdcop_v11_fold0.zip',
    'norm_stats_path': '/opt/airflow/models/norm_stats_v11.json',
    'model_id': 'ppo_usdcop_v11_fold0',
    'model_version': 'v11.2',
    'fold_id': 0,
}

FEATURES_CONFIG = {
    'features': [
        'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
        'rsi_9', 'atr_pct', 'adx_14', 'bb_position',
        'dxy_z', 'dxy_change_1d', 'dxy_mom_5d',
        'vix_z', 'vix_regime', 'embi_z',
        'brent_change_1d', 'brent_vol_5d',
        'rate_spread', 'usdmxn_ret_1h',
        'hour_sin', 'hour_cos'
    ],
    'obs_dim': 20,  # 18 features + 2 state vars
}

TRADING_CONFIG = {
    'initial_equity': 10000.0,
    'cost_per_trade_bps': 3.0,
    'position_threshold': 0.1,  # Cambio mínimo para considerar trade
    'market_hours_start': 8,
    'market_hours_end': 13,  # 12:55 PM última barra
}

# Estadísticas de normalización (del entrenamiento V11)
NORM_STATS = {
    'log_ret_5m': {'mean': 2e-06, 'std': 0.001138},
    'log_ret_1h': {'mean': 2.3e-05, 'std': 0.003776},
    'log_ret_4h': {'mean': 5.2e-05, 'std': 0.007768},
    'rsi_9': {'mean': 49.27, 'std': 23.07},
    'atr_pct': {'mean': 0.062, 'std': 0.0446},
    'adx_14': {'mean': 32.01, 'std': 16.36},
    'bb_position': {'mean': 0.493, 'std': 0.292},
    'dxy_z': {'mean': 0.0089, 'std': 1.439},
    'dxy_change_1d': {'mean': 4.5e-05, 'std': 0.0101},
    'dxy_mom_5d': {'mean': 0.0004, 'std': 0.0215},
    'vix_z': {'mean': -0.131, 'std': 1.442},
    'vix_regime': {'mean': 0.805, 'std': 1.014},
    'embi_z': {'mean': -0.013, 'std': 1.377},
    'brent_change_1d': {'mean': 0.0025, 'std': 0.0463},
    'brent_vol_5d': {'mean': 0.003, 'std': 0.00186},
    'rate_spread': {'mean': -0.0326, 'std': 1.400},
    'usdmxn_ret_1h': {'mean': -1e-05, 'std': 0.00367},
    'hour_sin': {'mean': 0.464, 'std': 0.309},
    'hour_cos': {'mean': -0.810, 'std': 0.183},
}

default_args = {
    'owner': 'trading_rl',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'email_on_failure': False,
}


# ═══════════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Obtiene conexión a PostgreSQL"""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def utc_to_cot(dt_utc):
    """Convierte UTC a Colombia Time (COT = UTC-5)"""
    from datetime import timezone
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc - timedelta(hours=5)


def get_bar_number(dt_cot):
    """Calcula número de barra del día (1-60)"""
    if dt_cot.hour < 8:
        return 0
    minutes_since_open = (dt_cot.hour - 8) * 60 + dt_cot.minute
    return (minutes_since_open // 5) + 1


# ═══════════════════════════════════════════════════════════════════════════════════
# TASKS
# ═══════════════════════════════════════════════════════════════════════════════════

def check_market_hours(**context):
    """Verifica si estamos en horario de mercado"""
    now_utc = datetime.utcnow()
    now_cot = utc_to_cot(now_utc)

    # Verificar día de semana (Lun=0, Dom=6)
    if now_cot.weekday() >= 5:
        print(f"Fin de semana: {now_cot.strftime('%A')}")
        return 'skip_inference'

    # Verificar hora
    hour = now_cot.hour
    minute = now_cot.minute

    # Mercado: 8:00 - 12:55
    if hour < 8 or (hour >= 13):
        print(f"Fuera de horario de mercado: {now_cot.strftime('%H:%M')}")
        return 'skip_inference'

    # Última barra válida es 12:55
    if hour == 12 and minute > 55:
        print(f"Después de última barra: {now_cot.strftime('%H:%M')}")
        return 'skip_inference'

    bar_number = get_bar_number(now_cot)
    print(f"En horario de mercado: {now_cot.strftime('%H:%M')} COT, Barra #{bar_number}")

    context['ti'].xcom_push(key='execution_time_utc', value=now_utc.isoformat())
    context['ti'].xcom_push(key='execution_time_cot', value=now_cot.isoformat())
    context['ti'].xcom_push(key='bar_number', value=bar_number)
    context['ti'].xcom_push(key='session_date', value=now_cot.date().isoformat())

    return 'get_current_state'


def get_current_state(**context):
    """Obtiene estado actual del portfolio"""
    session_date = context['ti'].xcom_pull(key='session_date')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Obtener última posición conocida del día
        cur.execute("""
            SELECT
                position_after,
                equity_after,
                pnl_daily
            FROM dw.fact_agent_actions
            WHERE session_date = %s
            ORDER BY bar_number DESC
            LIMIT 1
        """, (session_date,))

        row = cur.fetchone()

        if row:
            current_position = row[0]
            current_equity = float(row[1]) if row[1] else TRADING_CONFIG['initial_equity']
            daily_pnl = float(row[2]) if row[2] else 0.0
        else:
            # Primera barra del día
            current_position = 0.0
            current_equity = TRADING_CONFIG['initial_equity']
            daily_pnl = 0.0

        state = {
            'position': current_position,
            'equity': current_equity,
            'daily_pnl': daily_pnl,
            'log_portfolio': np.log(current_equity)
        }

        print(f"Estado actual: position={current_position:.2f}, equity=${current_equity:.2f}")
        context['ti'].xcom_push(key='current_state', value=state)

    finally:
        cur.close()
        conn.close()


def fetch_latest_data(**context):
    """Obtiene últimos datos OHLCV y features"""
    execution_time = context['ti'].xcom_pull(key='execution_time_cot')
    dt_cot = datetime.fromisoformat(execution_time)

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Obtener última vela OHLCV
        cur.execute("""
            SELECT
                time,
                open, high, low, close,
                volume
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
            ORDER BY time DESC
            LIMIT 1
        """)

        ohlcv_row = cur.fetchone()

        if not ohlcv_row:
            raise ValueError("No hay datos OHLCV disponibles")

        # Obtener datos macro más recientes
        cur.execute("""
            SELECT
                dxy, dxy_z, dxy_change_1d, dxy_mom_5d,
                vix, vix_z, vix_regime,
                embi, embi_z,
                brent, brent_change_1d, brent_vol_5d,
                rate_spread,
                usdmxn
            FROM dw.fact_macro_realtime
            ORDER BY timestamp_utc DESC
            LIMIT 1
        """)

        macro_row = cur.fetchone()

        # Calcular features técnicas de la última vela
        cur.execute("""
            WITH recent_bars AS (
                SELECT
                    time,
                    close,
                    high,
                    low,
                    LAG(close, 1) OVER (ORDER BY time) as prev_close_1,
                    LAG(close, 12) OVER (ORDER BY time) as prev_close_12,
                    LAG(close, 48) OVER (ORDER BY time) as prev_close_48
                FROM usdcop_m5_ohlcv
                WHERE symbol = 'USD/COP'
                ORDER BY time DESC
                LIMIT 100
            )
            SELECT
                time,
                close,
                CASE WHEN prev_close_1 > 0 THEN LN(close / prev_close_1) ELSE 0 END as log_ret_5m,
                CASE WHEN prev_close_12 > 0 THEN LN(close / prev_close_12) ELSE 0 END as log_ret_1h,
                CASE WHEN prev_close_48 > 0 THEN LN(close / prev_close_48) ELSE 0 END as log_ret_4h
            FROM recent_bars
            ORDER BY time DESC
            LIMIT 1
        """)

        returns_row = cur.fetchone()

        # Construir diccionario de features
        data = {
            'timestamp': ohlcv_row[0],
            'close': float(ohlcv_row[4]),
            'raw_return_5m': float(returns_row[2]) if returns_row else 0.0,

            # Returns
            'log_ret_5m': float(returns_row[2]) if returns_row else 0.0,
            'log_ret_1h': float(returns_row[3]) if returns_row else 0.0,
            'log_ret_4h': float(returns_row[4]) if returns_row else 0.0,

            # Técnicos (usar defaults si no hay datos calculados)
            'rsi_9': 50.0,
            'atr_pct': 0.06,
            'adx_14': 30.0,
            'bb_position': 0.5,

            # Macro
            'dxy_z': float(macro_row[1]) if macro_row and macro_row[1] else 0.0,
            'dxy_change_1d': float(macro_row[2]) if macro_row and macro_row[2] else 0.0,
            'dxy_mom_5d': float(macro_row[3]) if macro_row and macro_row[3] else 0.0,
            'vix_z': float(macro_row[5]) if macro_row and macro_row[5] else 0.0,
            'vix_regime': float(macro_row[6]) if macro_row and macro_row[6] else 1.0,
            'embi_z': float(macro_row[8]) if macro_row and macro_row[8] else 0.0,
            'brent_change_1d': float(macro_row[10]) if macro_row and macro_row[10] else 0.0,
            'brent_vol_5d': float(macro_row[11]) if macro_row and macro_row[11] else 0.003,
            'rate_spread': float(macro_row[12]) if macro_row and macro_row[12] else 0.0,
            'usdmxn_ret_1h': 0.0,  # Calcular si se tiene histórico

            # Temporales
            'hour_sin': np.sin(2 * np.pi * dt_cot.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dt_cot.hour / 24),
        }

        print(f"Datos obtenidos: close={data['close']:.2f}, log_ret_5m={data['log_ret_5m']:.6f}")
        context['ti'].xcom_push(key='market_data', value=data)

    finally:
        cur.close()
        conn.close()


def normalize_observation(**context):
    """Normaliza features usando estadísticas del entrenamiento"""
    market_data = context['ti'].xcom_pull(key='market_data')
    current_state = context['ti'].xcom_pull(key='current_state')
    bar_number = context['ti'].xcom_pull(key='bar_number')

    obs_normalized = []

    # Normalizar cada feature
    for feat in FEATURES_CONFIG['features']:
        if feat in market_data and feat in NORM_STATS:
            val = market_data[feat]
            mean = NORM_STATS[feat]['mean']
            std = NORM_STATS[feat]['std']

            # Z-score normalization con clip
            if std > 0:
                normalized = (val - mean) / std
                normalized = np.clip(normalized, -5, 5)
            else:
                normalized = 0.0

            # Handle NaN
            if np.isnan(normalized):
                normalized = 0.0

            obs_normalized.append(float(normalized))
        else:
            obs_normalized.append(0.0)

    # Agregar variables de estado
    obs_normalized.append(float(current_state['position']))  # Posición actual
    obs_normalized.append(float(bar_number) / 60.0)  # Progreso del día

    observation = {
        'normalized': obs_normalized,
        'raw': market_data
    }

    print(f"Observación: {len(obs_normalized)} features, position={current_state['position']:.2f}")
    context['ti'].xcom_push(key='observation', value=observation)


def run_inference(**context):
    """Ejecuta inferencia del modelo PPO"""
    import time as time_module

    observation = context['ti'].xcom_pull(key='observation')
    obs_array = np.array(observation['normalized'], dtype=np.float32)

    start_time = time_module.time()

    try:
        # Intentar cargar modelo real
        from stable_baselines3 import PPO

        if os.path.exists(MODEL_CONFIG['model_path']):
            model = PPO.load(MODEL_CONFIG['model_path'])
            action, _ = model.predict(obs_array, deterministic=True)
            action_value = float(action[0])
            inference_source = 'ppo_model'
        else:
            # Modelo no disponible - usar fallback simple
            action_value = _simple_momentum_signal(observation['raw'])
            inference_source = 'fallback_momentum'

    except ImportError:
        # stable_baselines3 no disponible
        action_value = _simple_momentum_signal(observation['raw'])
        inference_source = 'fallback_momentum'

    latency_ms = int((time_module.time() - start_time) * 1000)

    # Discretizar acción
    if action_value > 0.3:
        action_discretized = 'LONG'
    elif action_value < -0.3:
        action_discretized = 'SHORT'
    else:
        action_discretized = 'HOLD'

    # Calcular confianza
    confidence = min(abs(action_value), 1.0)

    result = {
        'action_raw': action_value,
        'action_discretized': action_discretized,
        'confidence': confidence,
        'latency_ms': latency_ms,
        'inference_source': inference_source
    }

    print(f"Inferencia: action={action_value:.3f} ({action_discretized}), "
          f"conf={confidence:.2f}, latency={latency_ms}ms")

    context['ti'].xcom_push(key='inference_result', value=result)


def _simple_momentum_signal(data):
    """Fallback: señal simple basada en momentum"""
    log_ret_5m = data.get('log_ret_5m', 0)
    log_ret_1h = data.get('log_ret_1h', 0)

    # Combinar momentum de diferentes timeframes
    signal = 0.4 * np.sign(log_ret_5m) * min(abs(log_ret_5m) * 100, 1) + \
             0.6 * np.sign(log_ret_1h) * min(abs(log_ret_1h) * 50, 1)

    return np.clip(signal, -1, 1)


def calculate_portfolio_update(**context):
    """Calcula actualización del portfolio"""
    inference_result = context['ti'].xcom_pull(key='inference_result')
    current_state = context['ti'].xcom_pull(key='current_state')
    market_data = context['ti'].xcom_pull(key='market_data')

    prev_position = current_state['position']
    new_position = inference_result['action_raw']
    raw_return = market_data['raw_return_5m']

    # Calcular costo de transacción
    position_change = abs(new_position - prev_position)
    transaction_cost_bps = 0.0

    if position_change > TRADING_CONFIG['position_threshold']:
        transaction_cost_bps = position_change * TRADING_CONFIG['cost_per_trade_bps']

    # Actualizar portfolio en log-space (numéricamente estable)
    log_portfolio = current_state['log_portfolio']

    # Costo de transacción
    if transaction_cost_bps > 0:
        cost_factor = 1.0 - (transaction_cost_bps / 10000)
        log_portfolio += np.log(max(cost_factor, 0.001))

    # Retorno del mercado * posición
    log_portfolio += prev_position * raw_return

    # Convertir a equity
    new_equity = np.exp(log_portfolio)
    pnl_bar = new_equity - current_state['equity']

    portfolio_update = {
        'position_before': prev_position,
        'position_after': new_position,
        'position_change': new_position - prev_position,
        'equity_before': current_state['equity'],
        'equity_after': new_equity,
        'log_portfolio': log_portfolio,
        'pnl_bar': pnl_bar,
        'pnl_daily': current_state['daily_pnl'] + pnl_bar,
        'transaction_cost_bps': transaction_cost_bps,
        'raw_return': raw_return
    }

    print(f"Portfolio: ${current_state['equity']:.2f} -> ${new_equity:.2f}, "
          f"PnL=${pnl_bar:.2f}")

    context['ti'].xcom_push(key='portfolio_update', value=portfolio_update)


def store_inference(**context):
    """Almacena inferencia en fact_rl_inference"""
    execution_time_utc = context['ti'].xcom_pull(key='execution_time_utc')
    execution_time_cot = context['ti'].xcom_pull(key='execution_time_cot')
    observation = context['ti'].xcom_pull(key='observation')
    inference_result = context['ti'].xcom_pull(key='inference_result')
    portfolio_update = context['ti'].xcom_pull(key='portfolio_update')
    market_data = context['ti'].xcom_pull(key='market_data')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO dw.fact_rl_inference (
                timestamp_utc, timestamp_cot,
                model_id, model_version, fold_id,
                observation, observation_raw,
                action_raw, action_discretized, confidence,
                close_price, raw_return_5m,
                position_before, portfolio_value_before, log_portfolio_before,
                position_after, portfolio_value_after, log_portfolio_after,
                position_change, transaction_cost_bps,
                latency_ms, inference_source, dag_run_id
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s
            ) RETURNING inference_id
        """, (
            execution_time_utc, execution_time_cot,
            MODEL_CONFIG['model_id'], MODEL_CONFIG['model_version'], MODEL_CONFIG['fold_id'],
            observation['normalized'], json.dumps(observation['raw']),
            inference_result['action_raw'], inference_result['action_discretized'],
            inference_result['confidence'],
            market_data['close'], market_data['raw_return_5m'],
            portfolio_update['position_before'], portfolio_update['equity_before'],
            np.log(portfolio_update['equity_before']),
            portfolio_update['position_after'], portfolio_update['equity_after'],
            portfolio_update['log_portfolio'],
            portfolio_update['position_change'], portfolio_update['transaction_cost_bps'],
            inference_result['latency_ms'], inference_result['inference_source'],
            context['dag_run'].run_id
        ))

        inference_id = cur.fetchone()[0]
        conn.commit()

        print(f"Inferencia almacenada: ID={inference_id}")
        context['ti'].xcom_push(key='inference_id', value=inference_id)

    finally:
        cur.close()
        conn.close()


def store_agent_action(**context):
    """Almacena acción en fact_agent_actions"""
    execution_time_utc = context['ti'].xcom_pull(key='execution_time_utc')
    execution_time_cot = context['ti'].xcom_pull(key='execution_time_cot')
    session_date = context['ti'].xcom_pull(key='session_date')
    bar_number = context['ti'].xcom_pull(key='bar_number')
    inference_result = context['ti'].xcom_pull(key='inference_result')
    portfolio_update = context['ti'].xcom_pull(key='portfolio_update')
    market_data = context['ti'].xcom_pull(key='market_data')
    inference_id = context['ti'].xcom_pull(key='inference_id')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # El trigger auto-completará action_type, side, marker_type, marker_color
        cur.execute("""
            INSERT INTO dw.fact_agent_actions (
                timestamp_utc, timestamp_cot,
                session_date, bar_number,
                price_at_action,
                position_before, position_after,
                equity_before, equity_after,
                pnl_action, pnl_daily,
                model_confidence, model_id,
                inference_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING action_id
        """, (
            execution_time_utc, execution_time_cot,
            session_date, bar_number,
            market_data['close'],
            portfolio_update['position_before'], portfolio_update['position_after'],
            portfolio_update['equity_before'], portfolio_update['equity_after'],
            portfolio_update['pnl_bar'], portfolio_update['pnl_daily'],
            inference_result['confidence'], MODEL_CONFIG['model_id'],
            inference_id
        ))

        action_id = cur.fetchone()[0]
        conn.commit()

        print(f"Acción almacenada: ID={action_id}")

    finally:
        cur.close()
        conn.close()


def update_equity_curve(**context):
    """Actualiza curva de equity en tiempo real"""
    execution_time_utc = context['ti'].xcom_pull(key='execution_time_utc')
    execution_time_cot = context['ti'].xcom_pull(key='execution_time_cot')
    session_date = context['ti'].xcom_pull(key='session_date')
    bar_number = context['ti'].xcom_pull(key='bar_number')
    portfolio_update = context['ti'].xcom_pull(key='portfolio_update')
    market_data = context['ti'].xcom_pull(key='market_data')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Obtener high water mark
        cur.execute("""
            SELECT COALESCE(MAX(equity_value), %s) as hwm
            FROM dw.fact_equity_curve_realtime
            WHERE session_date = %s
        """, (TRADING_CONFIG['initial_equity'], session_date))

        hwm = float(cur.fetchone()[0])
        hwm = max(hwm, portfolio_update['equity_after'])

        # Calcular drawdown
        current_dd_pct = (hwm - portfolio_update['equity_after']) / hwm if hwm > 0 else 0

        # Calcular retornos
        return_daily_pct = (portfolio_update['equity_after'] - TRADING_CONFIG['initial_equity']) / \
                           TRADING_CONFIG['initial_equity'] * 100

        # Determinar side
        pos = portfolio_update['position_after']
        side = 'LONG' if pos > 0.1 else ('SHORT' if pos < -0.1 else None)

        cur.execute("""
            INSERT INTO dw.fact_equity_curve_realtime (
                timestamp_utc, timestamp_cot, session_date, bar_number,
                equity_value, log_equity,
                unrealized_pnl, realized_pnl,
                return_bar_pct, return_daily_pct,
                high_water_mark, current_drawdown_pct,
                current_position, position_side,
                market_price, market_return_bar,
                model_id
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s
            )
        """, (
            execution_time_utc, execution_time_cot, session_date, bar_number,
            portfolio_update['equity_after'], portfolio_update['log_portfolio'],
            0, portfolio_update['pnl_daily'],
            portfolio_update['pnl_bar'] / portfolio_update['equity_before'] * 100 if portfolio_update['equity_before'] > 0 else 0,
            return_daily_pct,
            hwm, current_dd_pct,
            portfolio_update['position_after'], side,
            market_data['close'], market_data['raw_return_5m'] * 100,
            MODEL_CONFIG['model_id']
        ))

        conn.commit()
        print(f"Equity curve actualizada: ${portfolio_update['equity_after']:.2f}, DD={current_dd_pct:.2%}")

    finally:
        cur.close()
        conn.close()


def update_session_performance(**context):
    """Actualiza métricas de sesión"""
    session_date = context['ti'].xcom_pull(key='session_date')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT dw.update_session_metrics(%s)", (session_date,))
        conn.commit()
        print(f"Métricas de sesión actualizadas: {session_date}")

    finally:
        cur.close()
        conn.close()


def check_alerts(**context):
    """Verifica y genera alertas si es necesario"""
    portfolio_update = context['ti'].xcom_pull(key='portfolio_update')
    inference_result = context['ti'].xcom_pull(key='inference_result')
    session_date = context['ti'].xcom_pull(key='session_date')
    bar_number = context['ti'].xcom_pull(key='bar_number')

    alerts = []

    # Alta latencia
    if inference_result['latency_ms'] > 1000:
        alerts.append({
            'type': 'HIGH_LATENCY',
            'severity': 'WARNING',
            'message': f"Latencia de inferencia alta: {inference_result['latency_ms']}ms"
        })

    # Drawdown warning
    daily_return_pct = (portfolio_update['equity_after'] - TRADING_CONFIG['initial_equity']) / \
                       TRADING_CONFIG['initial_equity']
    if daily_return_pct < -0.02:  # -2%
        alerts.append({
            'type': 'DRAWDOWN_WARNING',
            'severity': 'WARNING',
            'message': f"Drawdown intraday alto: {daily_return_pct:.2%}"
        })

    if daily_return_pct < -0.05:  # -5%
        alerts.append({
            'type': 'DRAWDOWN_WARNING',
            'severity': 'ERROR',
            'message': f"Drawdown intraday crítico: {daily_return_pct:.2%}"
        })

    # Guardar alertas
    if alerts:
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            for alert in alerts:
                cur.execute("""
                    INSERT INTO dw.fact_inference_alerts (
                        alert_type, severity, message,
                        model_id, session_date, bar_number,
                        details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    alert['type'], alert['severity'], alert['message'],
                    MODEL_CONFIG['model_id'], session_date, bar_number,
                    json.dumps({'portfolio': portfolio_update, 'inference': inference_result})
                ))

            conn.commit()
            print(f"Alertas generadas: {len(alerts)}")

        finally:
            cur.close()
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Inferencia RL en tiempo real cada 5 minutos durante horario de mercado',
    schedule_interval='*/5 8-12 * * 1-5',  # Cada 5min, 8am-12pm COT, Lun-Vie
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['realtime', 'inference', 'rl', 'l5', 'production']
) as dag:

    start = EmptyOperator(task_id='start')

    # Branch: verificar horario de mercado
    check_hours = BranchPythonOperator(
        task_id='check_market_hours',
        python_callable=check_market_hours
    )

    skip = EmptyOperator(task_id='skip_inference')

    # Pipeline principal
    get_state = PythonOperator(
        task_id='get_current_state',
        python_callable=get_current_state
    )

    fetch_data = PythonOperator(
        task_id='fetch_latest_data',
        python_callable=fetch_latest_data
    )

    normalize = PythonOperator(
        task_id='normalize_observation',
        python_callable=normalize_observation
    )

    inference = PythonOperator(
        task_id='run_inference',
        python_callable=run_inference
    )

    calc_portfolio = PythonOperator(
        task_id='calculate_portfolio_update',
        python_callable=calculate_portfolio_update
    )

    store_inf = PythonOperator(
        task_id='store_inference',
        python_callable=store_inference
    )

    store_action = PythonOperator(
        task_id='store_agent_action',
        python_callable=store_agent_action
    )

    update_equity = PythonOperator(
        task_id='update_equity_curve',
        python_callable=update_equity_curve
    )

    update_session = PythonOperator(
        task_id='update_session_performance',
        python_callable=update_session_performance
    )

    check_alert = PythonOperator(
        task_id='check_alerts',
        python_callable=check_alerts
    )

    end = EmptyOperator(task_id='end', trigger_rule='none_failed_min_one_success')

    # Flujo
    start >> check_hours

    check_hours >> skip >> end
    check_hours >> get_state >> fetch_data >> normalize >> inference >> calc_portfolio

    calc_portfolio >> store_inf >> store_action
    calc_portfolio >> update_equity

    [store_action, update_equity] >> update_session >> check_alert >> end
