"""
SCRIPT 03b - Generacion de 10 Datasets RL para USD/COP - VERSION DIARIA
========================================================================
Genera los mismos 10 datasets pero con frecuencia DIARIA en lugar de 5 minutos.

Datasets:
  BASELINE (DS1-DS5):
    1. DS1_MINIMAL        - Baseline para validar pipeline
    2. DS2_TECHNICAL_MTF  - Multi-timeframe tecnico
    3. DS3_MACRO_CORE     - RECOMENDADO para produccion
    4. DS4_COST_AWARE     - Filtros anti-overtrading
    5. DS5_REGIME         - Deteccion de regimen

  ESPECIALIZADOS (DS6-DS10):
    6. DS6_CARRY_TRADE    - Diferenciales de tasas
    7. DS7_COMMODITY_BASKET - Commodities Colombia
    8. DS8_RISK_SENTIMENT - Risk-On/Risk-Off
    9. DS9_FED_WATCH      - Politica monetaria FED
   10. DS10_FLOWS_FUNDAMENTALS - Balanza de pagos y flujos

DIFERENCIAS vs version 5min:
  - Frecuencia: Diaria en lugar de 5 minutos
  - OHLC de USD_COP Historical Data.csv (Investing.com)
  - Macro de MACRO_DAILY_CONSOLIDATED.csv
  - Periodos ajustados (5 dias = 1 semana, 22 dias = 1 mes, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION GLOBAL
# =============================================================================
BASE_PATH = Path(__file__).parent  # 06_rl_dataset_builder/
PIPELINE_PATH = BASE_PATH.parent  # pipeline/

# Input paths (nueva estructura)
RAW_SOURCES = PIPELINE_PATH / "01_sources"
RESAMPLING_OUTPUT = PIPELINE_PATH / "05_resampling" / "output"

# Fallback a ubicacion anterior
if not RAW_SOURCES.exists():
    RAW_SOURCES = PIPELINE_PATH / "00_raw_sources"
if not RESAMPLING_OUTPUT.exists():
    RESAMPLING_OUTPUT = PIPELINE_PATH / "03_processing" / "intermediate"

# Input files
USDCOP_DAILY = RAW_SOURCES / "16_usdcop_historical" / "USD_COP Historical Data.csv"
if not USDCOP_DAILY.exists():
    USDCOP_DAILY = PIPELINE_PATH / "00_raw_sources" / "usdcop_daily_historical" / "USD_COP Historical Data.csv"
MACRO_DAILY = RESAMPLING_OUTPUT / "MACRO_DAILY_CONSOLIDATED.csv"

# Output
OUTPUT_RL_DAILY = PIPELINE_PATH / "07_output" / "datasets_daily"
OUTPUT_RL_DAILY.mkdir(parents=True, exist_ok=True)

# Fecha de corte (evita warmup)
START_DATE = '2020-03-01'

# Parametros tecnicos (ajustados para frecuencia diaria)
RSI_PERIOD = 9
ATR_PERIOD = 10
ADX_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 10
BB_STD = 2.0
MACD_FAST = 5
MACD_SLOW = 13
MACD_SIGNAL = 5
ZSCORE_WINDOW = 50  # 50 dias

# VIX regime thresholds
VIX_THRESHOLDS = [20, 25, 30]

print("=" * 80)
print("GENERACION DE 10 DATASETS RL PARA USD/COP - VERSION DIARIA")
print("=" * 80)
print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Fecha corte: {START_DATE}")
print(f"Output: {OUTPUT_RL_DAILY}")


# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print("\n" + "-" * 80)
print("1. CARGANDO DATOS")
print("-" * 80)

# OHLCV Diario
df_ohlcv = pd.read_csv(USDCOP_DAILY)
# Limpiar nombres de columnas
df_ohlcv.columns = df_ohlcv.columns.str.strip().str.replace('"', '')

# Parsear fecha (formato MM/DD/YYYY)
df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'], format='%m/%d/%Y')
df_ohlcv = df_ohlcv.rename(columns={'Date': 'date'})

# Limpiar columnas numericas (formato con comas)
for col in ['Price', 'Open', 'High', 'Low']:
    df_ohlcv[col] = df_ohlcv[col].astype(str).str.replace(',', '').astype(float)

df_ohlcv = df_ohlcv.rename(columns={
    'Price': 'close',
    'Open': 'open',
    'High': 'high',
    'Low': 'low'
})

df_ohlcv = df_ohlcv[['date', 'open', 'high', 'low', 'close']].copy()
df_ohlcv = df_ohlcv.sort_values('date').reset_index(drop=True)

print(f"   OHLCV Diario: {len(df_ohlcv):,} registros")
print(f"   Rango: {df_ohlcv['date'].min().date()} a {df_ohlcv['date'].max().date()}")

# Macro consolidado diario
df_macro = pd.read_csv(MACRO_DAILY)
df_macro['fecha'] = pd.to_datetime(df_macro['fecha'])
df_macro = df_macro.rename(columns={'fecha': 'date'})

print(f"   Macro Diario: {len(df_macro):,} registros, {len(df_macro.columns)} columnas")


# =============================================================================
# 2. MERGE OHLCV + MACRO
# =============================================================================
print("\n" + "-" * 80)
print("2. MERGE OHLCV + MACRO")
print("-" * 80)

# Columnas macro disponibles
macro_cols_to_use = [c for c in df_macro.columns if c != 'date']
print(f"   Columnas macro disponibles: {len(macro_cols_to_use)}")

df = pd.merge(
    df_ohlcv.sort_values('date'),
    df_macro.sort_values('date'),
    on='date',
    how='left'
)

print(f"   Despues de merge: {len(df):,} registros")


# =============================================================================
# 3. FUNCIONES DE INDICADORES
# =============================================================================

def calc_log_return(series, periods=1):
    """Log returns"""
    return np.log(series / series.shift(periods))

def calc_rsi(close, period=14):
    """RSI - Relative Strength Index (0-100)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    """ATR - Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_atr_pct(high, low, close, period=14):
    """ATR como porcentaje del precio"""
    atr = calc_atr(high, low, close, period)
    return (atr / close) * 100

def calc_adx(high, low, close, period=14):
    """ADX - Average Directional Index (0-100)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window=period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window=period).mean() / (atr + 1e-10)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    return adx

def calc_bollinger_position(close, period=20, std=2):
    """Posicion en Bollinger Bands (0-1)"""
    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    position = (close - lower) / (upper - lower + 1e-10)
    return position.clip(0, 1)

def calc_macd_histogram(close, fast=12, slow=26, signal=9):
    """MACD Histogram"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def calc_sma_ratio(close, period=20):
    """Ratio close / SMA - 1"""
    sma = close.rolling(window=period).mean()
    return (close / sma) - 1

def z_score_rolling(series, window=50, clip_val=4.0):
    """Rolling Z-score con clipping"""
    mean = series.rolling(window=window, min_periods=10).mean()
    std = series.rolling(window=window, min_periods=10).std()
    z = (series - mean) / (std + 1e-10)
    return z.clip(-clip_val, clip_val)

def pct_change_safe(series, periods=1):
    """Cambio porcentual seguro"""
    return series.pct_change(periods).clip(-0.2, 0.2)

def rolling_volatility(series, window):
    """Volatilidad rolling (std de retornos)"""
    returns = series.pct_change()
    return returns.rolling(window).std()

def calc_momentum(series, period=5):
    """Momentum: retorno acumulado en N periodos"""
    return (series / series.shift(period)) - 1

def calc_atr_percentile(atr, window=50):
    """Percentil del ATR actual vs ventana historica"""
    def percentile_rank(x):
        if len(x) < 2:
            return 0.5
        return (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
    return atr.rolling(window).apply(percentile_rank, raw=False)

def encode_cyclical(value, max_value):
    """Codificacion ciclica (sin/cos)"""
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


# =============================================================================
# 4. CALCULAR TODOS LOS FEATURES BASE
# =============================================================================
print("\n" + "-" * 80)
print("3. CALCULANDO FEATURES BASE")
print("-" * 80)

# --- RETORNOS (ajustados para diario) ---
print("   Calculando retornos...")
df['log_ret_1d'] = calc_log_return(df['close'], 1).clip(-0.1, 0.1)
df['log_ret_5d'] = calc_log_return(df['close'], 5).clip(-0.15, 0.15)  # 1 semana
df['log_ret_22d'] = calc_log_return(df['close'], 22).clip(-0.25, 0.25)  # 1 mes

# Lags
df['ret_lag_1'] = df['log_ret_1d'].shift(1)
df['ret_lag_5'] = df['log_ret_1d'].shift(5)

# Momentum
df['momentum_5d'] = df['log_ret_1d'].rolling(5).sum()
df['momentum_22d'] = df['log_ret_1d'].rolling(22).sum()

# --- TECNICOS ---
print("   Calculando indicadores tecnicos...")
df['rsi_9'] = calc_rsi(df['close'], RSI_PERIOD)
df['rsi_14'] = calc_rsi(df['close'], 14)
df['atr'] = calc_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
df['atr_pct'] = calc_atr_pct(df['high'], df['low'], df['close'], ATR_PERIOD)
df['adx_14'] = calc_adx(df['high'], df['low'], df['close'], ADX_PERIOD)
df['bb_position'] = calc_bollinger_position(df['close'], BB_PERIOD, BB_STD)
df['sma_ratio'] = calc_sma_ratio(df['close'], SMA_PERIOD)
df['macd_hist'] = calc_macd_histogram(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

# RSI multi-timeframe (semanal y mensual)
df['rsi_weekly'] = calc_rsi(df['close'].rolling(5).mean(), RSI_PERIOD)

# ATR percentile
df['atr_percentile'] = calc_atr_percentile(df['atr'], ZSCORE_WINDOW)

# RSI extremos
df['rsi_extreme'] = ((df['rsi_9'] < 25) | (df['rsi_9'] > 75)).astype(float)

# ADX strong
df['adx_strong'] = (df['adx_14'] > 25).astype(float)

# Volatility regime
df['vol_regime'] = pd.cut(
    df['atr_percentile'],
    bins=[-0.01, 0.25, 0.75, 1.01],
    labels=[0, 1, 2]
).astype(float)

# Retorno ajustado por ATR
df['ret_atr_adj'] = df['log_ret_1d'] / (df['atr_pct'] / 100 + 1e-10)
df['ret_atr_adj'] = df['ret_atr_adj'].clip(-3, 3)

# --- MACRO FEATURES ---
print("   Calculando features macro...")

# Mapeo de nombres de columnas macro
macro_mapping = {
    'FXRT_INDEX_DXY_USA_D_DXY': 'dxy',
    'VOLT_VIX_USA_D_VIX': 'vix',
    'COMM_OIL_BRENT_GLB_D_BRENT': 'brent',
    'COMM_OIL_WTI_GLB_D_WTI': 'wti',
    'COMM_AGRI_COFFEE_GLB_D_COFFEE': 'coffee',
    'COMM_METAL_GOLD_GLB_D_GOLD': 'gold',
    'CRSK_SPREAD_EMBI_COL_D_EMBI': 'embi',
    'FINC_BOND_YIELD10Y_USA_D_UST10Y': 'ust10y',
    'FINC_BOND_YIELD2Y_USA_D_DGS2': 'ust2y',
    'FINC_BOND_YIELD10Y_COL_D_COL10Y': 'col10y',
    'FINC_BOND_YIELD5Y_COL_D_COL5Y': 'col5y',
    'FINC_RATE_IBR_OVERNIGHT_COL_D_IBR': 'ibr',
    'EQTY_INDEX_COLCAP_COL_D_COLCAP': 'colcap',
    'FXRT_SPOT_USDMXN_MEX_D_USDMXN': 'usdmxn',
    'FXRT_SPOT_USDCLP_CHL_D_USDCLP': 'usdclp',
    'POLR_POLICY_RATE_COL_M_TPM': 'tpm',
    'POLR_FED_FUNDS_USA_M_FEDFUNDS': 'fedfunds',
    'INFL_CPI_ALL_USA_M_CPIAUCSL': 'cpi_usa',
    'INFL_PCE_USA_M_PCEPI': 'pce_usa',
    'LABR_UNEMPLOYMENT_USA_M_UNRATE': 'unemployment',
    'FTRD_TERMS_TRADE_COL_M_TOT': 'terms_trade',
    'FTRD_EXPORTS_TOTAL_COL_M_EXPUSD': 'exports',
    'FTRD_IMPORTS_TOTAL_COL_M_IMPUSD': 'imports',
    'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT': 'reserves',
    'FXRT_REER_BILATERAL_COL_M_ITCR': 'itcr',
    'RSBP_FDI_INFLOW_COL_Q_FDIIN_Q': 'fdi_inflow',
    'RSBP_CURRENT_ACCOUNT_COL_Q_CACCT_Q': 'current_account',
}

# Renombrar columnas macro existentes
for old_name, new_name in macro_mapping.items():
    if old_name in df.columns:
        df[new_name] = df[old_name].ffill()

# --- DXY ---
if 'dxy' in df.columns:
    df['dxy_z'] = z_score_rolling(df['dxy'], ZSCORE_WINDOW)
    df['dxy_change_1d'] = pct_change_safe(df['dxy'], 1)
    df['dxy_change_5d'] = pct_change_safe(df['dxy'], 5)

# --- VIX ---
if 'vix' in df.columns:
    df['vix_level'] = df['vix']
    df['vix_z'] = z_score_rolling(df['vix'], ZSCORE_WINDOW)
    df['vix_regime'] = pd.cut(
        df['vix'],
        bins=[0, VIX_THRESHOLDS[0], VIX_THRESHOLDS[1], VIX_THRESHOLDS[2], 100],
        labels=[0, 1, 2, 3]
    ).astype(float).ffill()

# --- BRENT ---
if 'brent' in df.columns:
    df['brent_z'] = z_score_rolling(df['brent'], ZSCORE_WINDOW)
    df['brent_change_1d'] = pct_change_safe(df['brent'], 1)
    df['brent_change_5d'] = pct_change_safe(df['brent'], 5)
    df['brent_vol_22d'] = rolling_volatility(df['brent'], 22)

# --- WTI ---
if 'wti' in df.columns:
    df['wti_z'] = z_score_rolling(df['wti'], ZSCORE_WINDOW)
    df['wti_change_5d'] = pct_change_safe(df['wti'], 5)

# --- GOLD ---
if 'gold' in df.columns:
    df['gold_z'] = z_score_rolling(df['gold'], ZSCORE_WINDOW)
    df['gold_change_5d'] = pct_change_safe(df['gold'], 5)

# --- COFFEE ---
if 'coffee' in df.columns:
    df['coffee_z'] = z_score_rolling(df['coffee'], ZSCORE_WINDOW)
    df['coffee_change_5d'] = pct_change_safe(df['coffee'], 5)

# --- EMBI ---
if 'embi' in df.columns:
    df['embi_z'] = z_score_rolling(df['embi'], ZSCORE_WINDOW)
    df['embi_change_5d'] = pct_change_safe(df['embi'], 5)

# --- TASAS USA ---
if 'ust10y' in df.columns:
    df['ust10y_z'] = z_score_rolling(df['ust10y'], ZSCORE_WINDOW)

if 'ust2y' in df.columns and 'ust10y' in df.columns:
    df['curve_slope'] = df['ust10y'] - df['ust2y']
    df['curve_slope_z'] = z_score_rolling(df['curve_slope'], ZSCORE_WINDOW)
    df['curve_inverted'] = (df['curve_slope'] < 0).astype(float)

# --- TASAS COLOMBIA ---
if 'col10y' in df.columns and 'ust10y' in df.columns:
    df['col_us_spread'] = df['col10y'] - df['ust10y']
    df['col_us_spread_z'] = z_score_rolling(df['col_us_spread'], ZSCORE_WINDOW)

if 'col10y' in df.columns and 'col5y' in df.columns:
    df['col_curve_slope'] = df['col10y'] - df['col5y']

# --- POLICY RATES ---
if 'tpm' in df.columns and 'fedfunds' in df.columns:
    df['policy_spread'] = df['tpm'] - df['fedfunds']
    df['policy_spread_z'] = z_score_rolling(df['policy_spread'], ZSCORE_WINDOW)
    df['carry_favorable'] = (df['policy_spread'] > 2.0).astype(float)

# --- CROSS PAIRS ---
if 'usdmxn' in df.columns:
    df['usdmxn_ret_1d'] = pct_change_safe(df['usdmxn'], 1)
    df['usdmxn_z'] = z_score_rolling(df['usdmxn'], ZSCORE_WINDOW)

if 'usdclp' in df.columns:
    df['usdclp_ret_1d'] = pct_change_safe(df['usdclp'], 1)
    df['usdclp_z'] = z_score_rolling(df['usdclp'], ZSCORE_WINDOW)

# --- COLCAP ---
if 'colcap' in df.columns:
    df['colcap_ret_1d'] = pct_change_safe(df['colcap'], 1)
    df['colcap_z'] = z_score_rolling(df['colcap'], ZSCORE_WINDOW)

# --- TRADE BALANCE ---
if 'exports' in df.columns and 'imports' in df.columns:
    df['trade_balance'] = df['exports'] - df['imports']
    df['trade_balance_z'] = z_score_rolling(df['trade_balance'], ZSCORE_WINDOW)

# --- TEMPORALES ---
print("   Calculando features temporales...")
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['dow_sin'], df['dow_cos'] = encode_cyclical(df['day_of_week'], 5)
df['month_sin'], df['month_cos'] = encode_cyclical(df['month'], 12)

# Filtrar por fecha
df = df[df['date'] >= START_DATE].copy()
df = df.dropna(subset=['close'])
df = df.reset_index(drop=True)

print(f"\n   Dataset final: {len(df):,} registros")
print(f"   Columnas totales: {len(df.columns)}")


# =============================================================================
# 5. DEFINIR COMPOSICION DE CADA DATASET
# =============================================================================

# Columnas base siempre incluidas
BASE_COLS = ['date', 'open', 'high', 'low', 'close']

# DS1: MINIMAL - Solo lo esencial
DS1_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'atr_pct', 'bb_position',
    'dxy_z', 'vix_level', 'brent_z', 'embi_z',
    'dow_sin', 'dow_cos'
]

# DS2: TECHNICAL MTF - Enfoque tecnico
DS2_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d', 'log_ret_22d',
    'ret_lag_1', 'ret_lag_5', 'momentum_5d', 'momentum_22d',
    'rsi_9', 'rsi_14', 'rsi_weekly',
    'atr_pct', 'atr_percentile',
    'adx_14', 'adx_strong',
    'bb_position', 'sma_ratio', 'macd_hist',
    'dow_sin', 'dow_cos'
]

# DS3: MACRO CORE - Recomendado para produccion
DS3_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'atr_pct', 'bb_position', 'sma_ratio',
    'dxy', 'dxy_z', 'dxy_change_1d',
    'vix', 'vix_level', 'vix_z', 'vix_regime',
    'brent', 'brent_z', 'brent_change_1d',
    'embi', 'embi_z',
    'ust10y', 'curve_slope', 'curve_slope_z',
    'usdmxn_ret_1d', 'usdclp_ret_1d',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
]

# DS4: COST AWARE - Anti-overtrading
DS4_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'rsi_extreme',
    'atr_pct', 'atr_percentile', 'vol_regime',
    'adx_14', 'adx_strong',
    'ret_atr_adj',
    'dxy_z', 'vix_level', 'vix_regime',
    'brent_z', 'embi_z',
    'dow_sin', 'dow_cos'
]

# DS5: REGIME - Deteccion de regimen
DS5_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d', 'log_ret_22d',
    'momentum_5d', 'momentum_22d',
    'rsi_9', 'rsi_extreme',
    'atr_pct', 'atr_percentile', 'vol_regime',
    'adx_14', 'adx_strong',
    'dxy_z', 'dxy_change_5d',
    'vix', 'vix_z', 'vix_regime',
    'brent_z', 'brent_vol_22d',
    'embi_z', 'embi_change_5d',
    'curve_slope', 'curve_slope_z', 'curve_inverted',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
]

# DS6: CARRY TRADE - Diferenciales de tasas
DS6_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'atr_pct',
    'col10y', 'col5y', 'ust10y', 'ust2y',
    'col_us_spread', 'col_us_spread_z',
    'col_curve_slope', 'curve_slope', 'curve_inverted',
    'tpm', 'fedfunds', 'policy_spread', 'policy_spread_z',
    'carry_favorable',
    'ibr',
    'dow_sin', 'dow_cos'
]

# DS7: COMMODITY BASKET - Commodities Colombia
DS7_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'atr_pct',
    'brent', 'brent_z', 'brent_change_1d', 'brent_change_5d',
    'wti', 'wti_z', 'wti_change_5d',
    'coffee', 'coffee_z', 'coffee_change_5d',
    'gold', 'gold_z', 'gold_change_5d',
    'terms_trade',
    'dxy_z', 'vix_regime',
    'dow_sin', 'dow_cos'
]

# DS8: RISK SENTIMENT - Risk-On/Risk-Off
DS8_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'atr_pct', 'vol_regime',
    'vix', 'vix_z', 'vix_regime',
    'embi', 'embi_z', 'embi_change_5d',
    'dxy', 'dxy_z', 'dxy_change_5d',
    'gold_z',
    'curve_slope', 'curve_inverted',
    'colcap_z', 'colcap_ret_1d',
    'usdmxn_z', 'usdclp_z',
    'dow_sin', 'dow_cos'
]

# DS9: FED WATCH - Politica monetaria FED
DS9_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'atr_pct',
    'fedfunds',
    'ust10y', 'ust2y', 'curve_slope', 'curve_slope_z', 'curve_inverted',
    'cpi_usa', 'pce_usa', 'unemployment',
    'dxy', 'dxy_z', 'dxy_change_5d',
    'vix_regime',
    'dow_sin', 'dow_cos'
]

# DS10: FLOWS FUNDAMENTALS - Balanza de pagos
DS10_COLS = BASE_COLS + [
    'log_ret_1d', 'log_ret_5d',
    'rsi_9', 'atr_pct',
    'exports', 'imports', 'trade_balance', 'trade_balance_z',
    'fdi_inflow', 'current_account',
    'reserves', 'itcr',
    'embi_z',
    'brent_z',
    'dxy_z',
    'dow_sin', 'dow_cos'
]

DATASETS = {
    'RL_DS1_MINIMAL_DAILY': DS1_COLS,
    'RL_DS2_TECHNICAL_MTF_DAILY': DS2_COLS,
    'RL_DS3_MACRO_CORE_DAILY': DS3_COLS,
    'RL_DS4_COST_AWARE_DAILY': DS4_COLS,
    'RL_DS5_REGIME_DAILY': DS5_COLS,
    'RL_DS6_CARRY_TRADE_DAILY': DS6_COLS,
    'RL_DS7_COMMODITY_BASKET_DAILY': DS7_COLS,
    'RL_DS8_RISK_SENTIMENT_DAILY': DS8_COLS,
    'RL_DS9_FED_WATCH_DAILY': DS9_COLS,
    'RL_DS10_FLOWS_FUNDAMENTALS_DAILY': DS10_COLS,
}


# =============================================================================
# 6. GENERAR Y GUARDAR DATASETS
# =============================================================================
print("\n" + "-" * 80)
print("4. GENERANDO DATASETS")
print("-" * 80)

for ds_name, cols in DATASETS.items():
    # Filtrar solo columnas existentes
    cols_exist = [c for c in cols if c in df.columns]
    cols_missing = [c for c in cols if c not in df.columns]

    df_ds = df[cols_exist].copy()

    # Dropear filas con nulls en columnas criticas
    critical_cols = ['close', 'log_ret_1d']
    critical_exist = [c for c in critical_cols if c in df_ds.columns]
    df_ds = df_ds.dropna(subset=critical_exist)

    # IMPORTANT: Only ffill allowed - bfill would create look-ahead bias
    df_ds = df_ds.ffill()

    # Guardar
    output_file = OUTPUT_RL_DAILY / f"{ds_name}.csv"
    df_ds.to_csv(output_file, index=False)

    print(f"   {ds_name}")
    print(f"      Filas: {len(df_ds):,} | Columnas: {len(df_ds.columns)}")
    print(f"      Rango: {df_ds['date'].min()} a {df_ds['date'].max()}")
    if cols_missing:
        print(f"      [WARN] Columnas faltantes: {cols_missing[:3]}...")


# =============================================================================
# 7. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN - DATASETS DIARIOS GENERADOS")
print("=" * 80)

import os
total_size = 0
for ds_name in DATASETS.keys():
    file_path = OUTPUT_RL_DAILY / f"{ds_name}.csv"
    if file_path.exists():
        size_kb = os.path.getsize(file_path) / 1024
        total_size += size_kb
        print(f"   {ds_name}.csv: {size_kb:.1f} KB")

print(f"\n   TOTAL: {total_size/1024:.1f} MB")
print(f"   Ubicacion: {OUTPUT_RL_DAILY}")
print("\n" + "=" * 80)
print("COMPLETADO")
print("=" * 80)
