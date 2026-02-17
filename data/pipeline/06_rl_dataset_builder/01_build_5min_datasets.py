"""
SCRIPT 03 - Generacion de 10 Datasets RL para USD/COP
======================================================
Version 3.0 - DATOS CRUDOS SIN FFILL
Version 15 - FIXED Z-SCORES (training-production parity)
Version 16 - UNIFIED LOADERS (SSOT from PostgreSQL/Parquet)

Genera 10 datasets optimizados para diferentes estrategias:

  BASELINE (DS1-DS5):
    1. DS1_MINIMAL        - Baseline para validar pipeline
    2. DS2_TECHNICAL_MTF  - Multi-timeframe tecnico puro
    3. DS3_MACRO_CORE     - RECOMENDADO para produccion ***
    4. DS4_COST_AWARE     - Filtros anti-overtrading
    5. DS5_REGIME         - Deteccion de regimen avanzado

  ESPECIALIZADOS (DS6-DS10):
    6. DS6_CARRY_TRADE    - Diferenciales de tasas (alto impacto esperado) **
    7. DS7_COMMODITY_BASKET - Commodities Colombia (petroleo, cafe, oro)
    8. DS8_RISK_SENTIMENT - Risk-On/Risk-Off para EM
    9. DS9_FED_WATCH      - Expectativas Fed y politica monetaria
   10. DS10_FLOWS_FUNDAMENTALS - Balanza de pagos y flujos

Estrategia de datos v3.0:
  - SIN FFILL: Datos crudos tal como vienen de la fuente
  - Los NaN se mantienen donde no hay datos (no se rellenan)
  - La expansion diario->5min ya viene hecha en MACRO_5MIN_CONSOLIDATED.csv
  - El filtrado de warmup NaN se hace al final con dropna()

SSOT Data Source (v16):
  - Primary: PostgreSQL (usdcop_m5_ohlcv, v_macro_unified)
  - Secondary: Parquet backups (seeds/latest/)
  - Legacy: CSV files (fallback)

Correcciones aplicadas:
  - Blacklist de columnas rotas
  - Normalizacion por tipo de feature
  - VIX regime con thresholds [20, 25, 30]
  - MACD corregido (signal=5)
  - Fecha corte: 2020-03-01
  - Nuevas variables macro para DS6-DS10
  - Removido ffill() de todas las columnas macro
"""

import pandas as pd
import numpy as np
import gzip
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import unified loaders
try:
    from src.data import UnifiedOHLCVLoader, UnifiedMacroLoader
    UNIFIED_LOADERS_AVAILABLE = True
except ImportError:
    UNIFIED_LOADERS_AVAILABLE = False
    print("[WARN] UnifiedLoaders not available - using legacy CSV loading")

# =============================================================================
# CONFIGURACION GLOBAL
# =============================================================================
BASE_PATH = Path(__file__).parent  # 06_rl_dataset_builder/
PIPELINE_PATH = BASE_PATH.parent  # pipeline/

# Input: datos resampleados de 05_resampling
INPUT_PATH = PIPELINE_PATH / "05_resampling" / "output"

# Fallback a ubicacion anterior
if not INPUT_PATH.exists():
    INPUT_PATH = PIPELINE_PATH / "03_processing" / "intermediate"

OUTPUT_V2 = INPUT_PATH  # Alias para compatibilidad
OUTPUT_RL = PIPELINE_PATH / "07_output" / "datasets_5min"
OUTPUT_RL.mkdir(parents=True, exist_ok=True)

# Backup OHLCV (fuera del pipeline - buscar en multiples ubicaciones)
PROJECT_ROOT = PIPELINE_PATH.parent.parent  # USDCOP-RL-Models/

# Buscar backup en multiples ubicaciones posibles
OHLCV_BACKUP_DIRS = [
    PROJECT_ROOT / "data" / "backups",
    PROJECT_ROOT / "backups" / "database",
    PROJECT_ROOT / "archive" / "backups" / "database",
]

ohlcv_files = []
for backup_dir in OHLCV_BACKUP_DIRS:
    if backup_dir.exists():
        ohlcv_files.extend(list(backup_dir.glob("usdcop_m5_ohlcv_*.csv.gz")))

# Encontrar el backup mas reciente
OHLCV_BACKUP = max(ohlcv_files, key=lambda x: x.stat().st_mtime) if ohlcv_files else PROJECT_ROOT / "data" / "backups" / "usdcop_m5_ohlcv_latest.csv.gz"

# Fecha de corte (evita warmup)
START_DATE = '2020-03-01'

# BLACKLIST - Columnas que NUNCA se incluyen
BLACKLIST = [
    'volume',           # 100% ceros (USD/COP es OTC)
    'obv_change_z',     # Depende de volume
    'macd_hist_z_old',  # Bug en calculo original
    'macd_1h_z_old',    # Bug en calculo original
    'dxy_cop_corr_z',   # Division por cero -> infinitos
    't_in_session',     # Sin varianza (bug)
    'risk_score_z',     # Outliers extremos
]

# FIXED NORMALIZATION STATS (v15 - aligned with feature_config.json)
# Calculated from training period 2020-03 to 2025-10
# These MUST match feature_config.json for training-production parity
FIXED_NORM_STATS = {
    'dxy': {'mean': 100.21, 'std': 5.60},
    'vix': {'mean': 21.16, 'std': 7.89},
    'embi': {'mean': 322.01, 'std': 62.68},
    'ust10y': {'mean': 2.97, 'std': 1.41},
    'ust2y': {'mean': 2.75, 'std': 1.88},
}

# Parametros tecnicos
RSI_PERIOD = 9
ATR_PERIOD = 10
ADX_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 10
BB_STD = 2.0
MACD_FAST = 5
MACD_SLOW = 13
MACD_SIGNAL = 5  # Corregido de 1 a 5
ZSCORE_WINDOW = 50

# VIX regime thresholds
VIX_THRESHOLDS = [20, 25, 30]  # calm/elevated/stress/crisis

# Horario liquido Colombia (COT)
LIQUID_HOURS = (8, 13)  # 8am - 1pm

print("=" * 80)
print("GENERACION DE 10 DATASETS RL PARA USD/COP v3.0 - DATOS CRUDOS")
print("=" * 80)
print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Fecha corte: {START_DATE}")
print(f"Output: {OUTPUT_RL}")


# =============================================================================
# 1. CARGAR DATOS - SSOT (PostgreSQL/Parquet) con fallback a CSV
# =============================================================================
print("\n" + "-" * 80)
print("1. CARGANDO DATOS")
print("-" * 80)

# Flag para modo de carga
USE_UNIFIED_LOADERS = UNIFIED_LOADERS_AVAILABLE

if USE_UNIFIED_LOADERS:
    try:
        print("   [SSOT] Using UnifiedLoaders (PostgreSQL/Parquet)...")

        # OHLCV from SSOT
        ohlcv_loader = UnifiedOHLCVLoader()
        df_ohlcv = ohlcv_loader.load_5min(START_DATE, "2026-12-31", filter_market_hours=True)

        # Normalize column names
        df_ohlcv = df_ohlcv.rename(columns={'time': 'datetime'})
        df_ohlcv = df_ohlcv[['datetime', 'open', 'high', 'low', 'close']].copy()

        print(f"   OHLCV (SSOT): {len(df_ohlcv):,} registros")
        print(f"   Rango: {df_ohlcv['datetime'].min()} a {df_ohlcv['datetime'].max()}")

        # Macro from SSOT (5-min resampled)
        macro_loader = UnifiedMacroLoader()
        df_macro = macro_loader.load_5min(START_DATE, "2026-12-31")

        # Rename columns to match expected format (uppercase)
        MACRO_COL_RENAMES = {
            'time': 'datetime',
            'dxy': 'FXRT_INDEX_DXY_USA_D_DXY',
            'vix': 'VOLT_VIX_USA_D_VIX',
            'embi': 'CRSK_SPREAD_EMBI_COL_D_EMBI',
            'brent': 'COMM_OIL_BRENT_GLB_D_BRENT',
            'wti': 'COMM_OIL_WTI_GLB_D_WTI',
            'gold': 'COMM_METAL_GOLD_GLB_D_GOLD',
            'coffee': 'COMM_AGRI_COFFEE_GLB_D_COFFEE',
            'ust10y': 'FINC_BOND_YIELD10Y_USA_D_UST10Y',
            'ust2y': 'FINC_BOND_YIELD2Y_USA_D_DGS2',
            'col10y': 'FINC_BOND_YIELD10Y_COL_D_COL10Y',
            'col5y': 'FINC_BOND_YIELD5Y_COL_D_COL5Y',
            'ibr': 'FINC_RATE_IBR_OVERNIGHT_COL_D_IBR',
            'tpm': 'POLR_POLICY_RATE_COL_M_TPM',
            'fedfunds': 'POLR_FED_FUNDS_USA_M_FEDFUNDS',
            'colcap': 'EQTY_INDEX_COLCAP_COL_D_COLCAP',
            'usdmxn': 'FXRT_SPOT_USDMXN_MEX_D_USDMXN',
            'usdclp': 'FXRT_SPOT_USDCLP_CHL_D_USDCLP',
            'cpi_usa': 'INFL_CPI_ALL_USA_M_CPIAUCSL',
            'pce': 'INFL_PCE_USA_M_PCEPI',
            'unemployment': 'LABR_UNEMPLOYMENT_USA_M_UNRATE',
            'terms_of_trade': 'FTRD_TERMS_TRADE_COL_M_TOT',
            'exports': 'FTRD_EXPORTS_TOTAL_COL_M_EXPUSD',
            'imports': 'FTRD_IMPORTS_TOTAL_COL_M_IMPUSD',
            'reserves': 'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT',
            'itcr': 'FXRT_REER_BILATERAL_COL_M_ITCR',
            'current_account': 'RSBP_CURRENT_ACCOUNT_COL_Q_CACCT',
            'rate_spread': 'RATE_SPREAD_COL_USA',
        }
        df_macro = df_macro.rename(columns=MACRO_COL_RENAMES)

        print(f"   Macro (SSOT): {len(df_macro):,} registros, {len(df_macro.columns)} columnas")

    except Exception as e:
        print(f"   [WARN] SSOT loading failed: {e}")
        print("   [WARN] Falling back to legacy CSV loading...")
        USE_UNIFIED_LOADERS = False

if not USE_UNIFIED_LOADERS:
    # Legacy CSV loading
    print("   [LEGACY] Using CSV file loading...")

    # OHLCV
    with gzip.open(OHLCV_BACKUP, 'rt') as f:
        df_ohlcv = pd.read_csv(f)

    df_ohlcv['time'] = pd.to_datetime(df_ohlcv['time'])
    df_ohlcv = df_ohlcv.sort_values('time').reset_index(drop=True)
    df_ohlcv = df_ohlcv.rename(columns={'time': 'datetime'})
    df_ohlcv = df_ohlcv[['datetime', 'open', 'high', 'low', 'close']].copy()

    print(f"   OHLCV (CSV): {len(df_ohlcv):,} registros")
    print(f"   Rango: {df_ohlcv['datetime'].min()} a {df_ohlcv['datetime'].max()}")

    # Macro consolidado
    df_macro = pd.read_csv(OUTPUT_V2 / "MACRO_5MIN_CONSOLIDATED.csv")
    df_macro['datetime_utc'] = pd.to_datetime(df_macro['datetime_utc'])
    df_macro['datetime'] = df_macro['datetime_utc'].dt.tz_localize(None)

    print(f"   Macro (CSV): {len(df_macro):,} registros, {len(df_macro.columns)} columnas")


# =============================================================================
# 2. MERGE OHLCV + MACRO
# =============================================================================
print("\n" + "-" * 80)
print("2. MERGE OHLCV + MACRO")
print("-" * 80)

# Remove timezone if present
if df_ohlcv['datetime'].dt.tz is not None:
    df_ohlcv['datetime'] = df_ohlcv['datetime'].dt.tz_localize(None)
if 'datetime' in df_macro.columns and df_macro['datetime'].dt.tz is not None:
    df_macro['datetime'] = df_macro['datetime'].dt.tz_localize(None)

# Columnas macro a incluir - EXPANDIDO para DS6-DS10
macro_cols_to_use = [
    'datetime',
    # === DIARIAS ===
    # Dolar y FX
    'FXRT_INDEX_DXY_USA_D_DXY',           # DXY
    'FXRT_SPOT_USDMXN_MEX_D_USDMXN',      # USD/MXN
    'FXRT_SPOT_USDCLP_CHL_D_USDCLP',      # USD/CLP
    # Volatilidad y riesgo
    'VOLT_VIX_USA_D_VIX',                 # VIX
    'CRSK_SPREAD_EMBI_COL_D_EMBI',        # EMBI Colombia
    # Commodities
    'COMM_OIL_BRENT_GLB_D_BRENT',         # Brent
    'COMM_OIL_WTI_GLB_D_WTI',             # WTI
    'COMM_AGRI_COFFEE_GLB_D_COFFEE',      # Coffee
    'COMM_METAL_GOLD_GLB_D_GOLD',         # Gold
    # Bonos USA
    'FINC_BOND_YIELD10Y_USA_D_UST10Y',    # UST 10Y
    'FINC_BOND_YIELD2Y_USA_D_DGS2',       # UST 2Y
    # Bonos Colombia (DS6)
    'FINC_BOND_YIELD10Y_COL_D_COL10Y',    # Colombia 10Y
    'FINC_BOND_YIELD5Y_COL_D_COL5Y',      # Colombia 5Y
    # Tasas interbancarias Colombia
    'FINC_RATE_IBR_OVERNIGHT_COL_D_IBR',  # IBR overnight
    # Equity Colombia
    'EQTY_INDEX_COLCAP_COL_D_COLCAP',     # COLCAP

    # === MENSUALES (forward-fill) ===
    # Politica monetaria
    'POLR_POLICY_RATE_COL_M_TPM',         # TPM Colombia
    'POLR_FED_FUNDS_USA_M_FEDFUNDS',      # Fed Funds
    # Inflacion USA
    'INFL_CPI_ALL_USA_M_CPIAUCSL',        # CPI USA
    'INFL_PCE_USA_M_PCEPI',               # PCE USA
    # Empleo USA
    'LABR_UNEMPLOYMENT_USA_M_UNRATE',     # Unemployment
    # Terminos de intercambio
    'FTRD_TERMS_TRADE_COL_M_TOT',         # Terms of Trade
    # Comercio exterior
    'FTRD_EXPORTS_TOTAL_COL_M_EXPUSD',    # Exports
    'FTRD_IMPORTS_TOTAL_COL_M_IMPUSD',    # Imports
    # Reservas y competitividad
    'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT',  # Reservas
    'FXRT_REER_BILATERAL_COL_M_ITCR',     # ITCR

    # === TRIMESTRALES (forward-fill) ===
    'RSBP_CURRENT_ACCOUNT_COL_Q_CACCT',   # Cuenta corriente
]

macro_cols_exist = [c for c in macro_cols_to_use if c in df_macro.columns]
df_macro_subset = df_macro[macro_cols_exist].copy()

print(f"   Columnas macro disponibles: {len(macro_cols_exist) - 1}")

# Merge OHLCV + Macro
if USE_UNIFIED_LOADERS and 'datetime' in df_macro_subset.columns:
    # SSOT mode: macro already has datetime aligned to 5-min grid
    # Use regular merge on datetime
    df = df_ohlcv.merge(df_macro_subset, on='datetime', how='left')
    print(f"   Merge mode: SSOT (direct join)")
else:
    # Legacy mode: use merge_asof for temporal alignment
    # P0-11 FIX: Remove tolerance to prevent data leakage
    df = pd.merge_asof(
        df_ohlcv.sort_values('datetime'),
        df_macro_subset.sort_values('datetime'),
        on='datetime',
        direction='backward'
        # NO tolerance - strict temporal ordering prevents data leakage
    )
    print(f"   Merge mode: Legacy (merge_asof)")

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
    """
    ADX - Average Directional Index (0-100)

    FIX v2: Use percentage-based ATR to avoid saturation issues.
    The original calculation saturated to ~100 because ATR in absolute
    terms for USDCOP (~4000) is very different from typical FX pairs (~1.0).

    Solution: Normalize DM by price level before calculating DI.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Use ATR as percentage of price to normalize across price levels
    atr = tr.rolling(window=period).mean()
    atr_pct = atr / close  # ATR as percentage

    # Minimum ATR to avoid division by zero (0.01% = 10 bps)
    atr_pct_min = 0.0001
    atr_pct_safe = atr_pct.clip(lower=atr_pct_min)

    up_move = high - high.shift()
    down_move = low.shift() - low

    # Directional movement as percentage of price
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move / close, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move / close, 0)

    # DI calculation with percentage-based values
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window=period).mean() / atr_pct_safe
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window=period).mean() / atr_pct_safe

    # Clamp DI values to valid range before DX calculation
    plus_di = plus_di.clip(0, 100)
    minus_di = minus_di.clip(0, 100)

    # DX with safe denominator
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum.clip(lower=1.0)

    adx = dx.rolling(window=period).mean()
    return adx.clip(0, 100)  # Ensure final output is bounded

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

def z_score_fixed(series, mean, std, clip_val=4.0):
    """
    Fixed Z-score normalization (v15).
    Uses constant mean/std from training period for training-production parity.

    CRITICAL: These stats MUST match feature_config.json
    """
    z = (series - mean) / (std + 1e-10)
    return z.clip(-clip_val, clip_val)

def pct_change_safe(series, periods=1):
    """Cambio porcentual seguro"""
    return series.pct_change(periods).clip(-0.1, 0.1)

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

def calc_volatility_pct(close, period=14, annualize=True, bars_per_day=48):
    """
    CLOSE-ONLY volatility (replaces ATR).
    Realized volatility as percentage of price.
    """
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=period).std()
    if annualize:
        vol = vol * np.sqrt(252 * bars_per_day)
    return vol * 100  # As percentage

def calc_trend_z(close, sma_period=50, clip_value=3.0):
    """
    CLOSE-ONLY trend indicator (replaces ADX).
    Position of price vs SMA normalized by rolling std.
    """
    sma = close.rolling(window=sma_period).mean()
    rolling_std = close.rolling(window=sma_period).std()
    trend_z = (close - sma) / (rolling_std + 1e-10)
    return trend_z.clip(-clip_value, clip_value)


# =============================================================================
# EXP-B-001: NEW FEATURE FUNCTIONS
# =============================================================================

def calc_rsi_21(close: pd.Series, period: int = 21) -> pd.Series:
    """
    EXP-B-001: RSI with 21-period for longer-term overbought/oversold.

    Args:
        close: Close price series
        period: RSI period (default 21)

    Returns:
        RSI values in [0, 100]
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_yield_curve_zscore(df: pd.DataFrame, window: int = 252 * 12) -> pd.Series:
    """
    EXP-B-001: Yield curve (10Y - 2Y) z-score for risk-on/off signal.

    Inverted yield curve (negative spread) historically indicates recession risk.

    Args:
        df: DataFrame with ust10y and ust2y columns
        window: Rolling window in 5-min bars (252 trading days * 12 hours/day)

    Returns:
        Rolling z-score of yield curve spread, clipped to [-4, 4]
    """
    if 'ust10y' in df.columns and 'ust2y' in df.columns:
        spread = df['ust10y'] - df['ust2y']
    else:
        return pd.Series(0.0, index=df.index)

    return z_score_rolling(spread, window=window, clip_val=4.0)


def calc_rate_spread_change(df: pd.DataFrame, periods: int = 288) -> pd.Series:
    """
    EXP-B-001: Rate spread momentum (col10y - ust10y) change.

    Captures carry trade momentum - widening spread is COP positive.

    Args:
        df: DataFrame with col10y and ust10y columns
        periods: Lookback period in 5-min bars (288 = 1 day)

    Returns:
        Percentage change of rate spread, clipped to [-0.1, 0.1]
    """
    if 'col10y' in df.columns and 'ust10y' in df.columns:
        spread = df['col10y'] - df['ust10y']
        return spread.pct_change(periods=periods).clip(-0.1, 0.1)
    return pd.Series(0.0, index=df.index)


def calculate_rate_spread_zscore(df: pd.DataFrame, window: int = 252 * 12 * 5) -> pd.Series:
    """
    Calculate rate spread as rolling z-score for stationarity.

    This fixes the distribution shift problem where rate_spread had:
    - Train (2020-2024): mean = +0.15
    - Test (2025): mean = -0.96 (shift of -1.11 std)

    By using rolling z-score, both train and test periods will have
    similar distributions (mean ~0, std ~1).

    Args:
        df: DataFrame with col10y and/or ust10y columns
        window: Rolling window in 5-min bars (252 trading days * 12 hours/day * 5 bars/hour)
                Default ~15,120 bars for 252 trading day window

    Returns:
        Rolling z-score of rate spread, clipped to [-4, 4]
    """
    # Calculate raw spread
    if 'col10y' in df.columns and 'ust10y' in df.columns:
        spread = df['col10y'] - df['ust10y']
    elif 'ust10y' in df.columns:
        # Fallback: use 10.0 - ust10y but with rolling normalization
        spread = 10.0 - df['ust10y']
    else:
        return pd.Series(0.0, index=df.index)

    # Apply rolling z-score (same method as dxy_z, vix_z, embi_z)
    # Use min_periods = window // 4 to allow calculation during warmup
    mean = spread.rolling(window=window, min_periods=window // 4).mean()
    std = spread.rolling(window=window, min_periods=window // 4).std()
    z = (spread - mean) / (std + 1e-10)

    return z.clip(-4.0, 4.0)

def encode_cyclical(value, max_value):
    """Codificacion ciclica (sin/cos)"""
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


# =============================================================================
# NORMALIZATION STATS (L2 -> L3 -> L4 Pipeline Integration)
# =============================================================================

def save_norm_stats(df: pd.DataFrame, output_dir: Path, dataset_name: str, feature_columns: list) -> Path:
    """
    Compute and save normalization stats from TRAINING data only.

    This ensures L2, L3, and L4 all use the SAME normalization statistics,
    fixing the critical bug where L4 was using unnormalized observations.

    Args:
        df: Full dataset with 'timestamp' column
        output_dir: Directory to save norm_stats.json
        dataset_name: Name for the output file (e.g., 'DS3_MACRO_CORE')
        feature_columns: List of feature column names to compute stats for

    Returns:
        Path to saved norm_stats.json
    """
    # Use only TRAINING data (before 2025-01-01) to compute stats
    # This prevents data leakage from test period
    TRAIN_END_DATE = '2025-01-01'

    if 'timestamp' in df.columns:
        time_col = 'timestamp'
    elif 'datetime' in df.columns:
        time_col = 'datetime'
    else:
        raise ValueError("Dataset must have 'timestamp' or 'datetime' column")

    # Filter to training period only
    train_df = df[df[time_col] < TRAIN_END_DATE].copy()

    if len(train_df) == 0:
        raise ValueError(f"No training data found before {TRAIN_END_DATE}")

    print(f"   Computing norm_stats from {len(train_df):,} training rows (before {TRAIN_END_DATE})")

    # Compute stats for each feature
    norm_stats = {}

    for feat in feature_columns:
        if feat not in train_df.columns:
            print(f"      [WARN] Feature '{feat}' not in dataset, skipping")
            continue

        values = train_df[feat].dropna()

        if len(values) == 0:
            print(f"      [WARN] Feature '{feat}' has all NaN values, using defaults")
            norm_stats[feat] = {
                "mean": 0.0,
                "std": 1.0,
                "min": -5.0,
                "max": 5.0,
            }
            continue

        mean_val = float(values.mean())
        std_val = float(values.std())
        min_val = float(values.min())
        max_val = float(values.max())

        # Prevent division by zero
        if std_val < 1e-8:
            std_val = 1.0

        norm_stats[feat] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        }

    # Compute feature order hash for contract validation
    feature_order_str = ",".join(sorted(feature_columns))
    feature_order_hash = hashlib.md5(feature_order_str.encode()).hexdigest()

    # Add metadata
    norm_stats["_meta"] = {
        "version": "1.0.0",
        "pipeline_stage": "L2",
        "train_rows": len(train_df),
        "train_end_date": TRAIN_END_DATE,
        "feature_count": len(feature_columns),
        "feature_order_hash": feature_order_hash,
        "created_at": datetime.now().isoformat(),
    }

    # Save to output directory
    stats_path = output_dir / f"{dataset_name}_norm_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print(f"   Saved norm_stats: {stats_path} ({len(norm_stats) - 1} features)")

    return stats_path


# =============================================================================
# 4. CALCULAR TODOS LOS FEATURES BASE
# =============================================================================
print("\n" + "-" * 80)
print("3. CALCULANDO FEATURES BASE")
print("-" * 80)

# --- RETORNOS ---
print("   Calculando retornos...")
# P0.1 FIX: Preserve RAW returns BEFORE any clipping/normalization
# raw_log_ret_5m is used for PnL calculation in L4 backtest
# log_ret_5m (clipped) is used as model input feature
df['raw_log_ret_5m'] = calc_log_return(df['close'], 1)
df['raw_log_ret_5m'] = df['raw_log_ret_5m'].fillna(0)  # First bar has no return

# Clipped versions for model features
df['log_ret_5m'] = df['raw_log_ret_5m'].clip(-0.05, 0.05)
df['log_ret_15m'] = calc_log_return(df['close'], 3).clip(-0.05, 0.05)
df['log_ret_1h'] = calc_log_return(df['close'], 12).clip(-0.05, 0.05)
df['log_ret_4h'] = calc_log_return(df['close'], 48).clip(-0.05, 0.05)

# Lags
df['ret_lag_1'] = df['log_ret_5m'].shift(1)
df['ret_lag_3'] = df['log_ret_5m'].shift(3)

# Momentum
df['momentum_6'] = df['log_ret_5m'].rolling(6).sum()

# --- TECNICOS ---
print("   Calculando indicadores tecnicos...")
df['rsi_9'] = calc_rsi(df['close'], RSI_PERIOD)
df['atr'] = calc_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
df['atr_pct'] = calc_atr_pct(df['high'], df['low'], df['close'], ATR_PERIOD)
df['adx_14'] = calc_adx(df['high'], df['low'], df['close'], ADX_PERIOD)
df['bb_position'] = calc_bollinger_position(df['close'], BB_PERIOD, BB_STD)
df['sma_ratio'] = calc_sma_ratio(df['close'], SMA_PERIOD)
df['macd_hist'] = calc_macd_histogram(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

# RSI multi-timeframe
df['rsi_9_15m'] = calc_rsi(df['close'].rolling(3).mean(), RSI_PERIOD)
df['rsi_9_1h'] = calc_rsi(df['close'].rolling(12).mean(), RSI_PERIOD)

# CLOSE-ONLY features (SSOT v2.0.0)
df['volatility_pct'] = calc_volatility_pct(df['close'], period=14, annualize=True, bars_per_day=48)
df['trend_z'] = calc_trend_z(df['close'], sma_period=50, clip_value=3.0)

# EXP-B-001: NEW FEATURES
print("   Calculating EXP-B-001 features...")
# 1. log_ret_1d: 1-day log return (288 bars)
df['log_ret_1d'] = calc_log_return(df['close'], 288).clip(-0.05, 0.05)
# 2. rsi_21: Longer-term RSI
df['rsi_21'] = calc_rsi_21(df['close'], period=21)

# ATR multi-timeframe
df['atr_pct_1h'] = calc_atr_pct(
    df['high'].rolling(12).max(),
    df['low'].rolling(12).min(),
    df['close'].rolling(12).mean(),
    ATR_PERIOD
)

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
df['ret_atr_adj'] = df['log_ret_5m'] / (df['atr_pct'] / 100 + 1e-10)
df['ret_atr_adj'] = df['ret_atr_adj'].clip(-3, 3)

# --- MACRO: DXY ---
print("   Calculando features macro (DXY)...")
if 'FXRT_INDEX_DXY_USA_D_DXY' in df.columns:
    df['dxy'] = df['FXRT_INDEX_DXY_USA_D_DXY']  # Sin ffill - datos crudos
    df['dxy_z'] = z_score_fixed(df['dxy'], **FIXED_NORM_STATS['dxy'])
    df['dxy_change_1d'] = pct_change_safe(df['dxy'], 12*24)
    df['dxy_mom_5d'] = calc_momentum(df['dxy'], 12*24*5)
    df['dxy_vol_5d'] = rolling_volatility(df['dxy'], 12*24*5)

# --- MACRO: VIX ---
print("   Calculando features macro (VIX)...")
if 'VOLT_VIX_USA_D_VIX' in df.columns:
    df['vix'] = df['VOLT_VIX_USA_D_VIX']  # Sin ffill - datos crudos
    df['vix_level'] = df['vix']  # Sin normalizar, interpretable
    df['vix_z'] = z_score_fixed(df['vix'], **FIXED_NORM_STATS['vix'])
    # VIX regime con thresholds correctos
    df['vix_regime'] = pd.cut(
        df['vix'],
        bins=[0, VIX_THRESHOLDS[0], VIX_THRESHOLDS[1], VIX_THRESHOLDS[2], 100],
        labels=[0, 1, 2, 3]  # calm/elevated/stress/crisis
    ).astype(float)

# --- MACRO: BRENT ---
print("   Calculando features macro (Brent)...")
if 'COMM_OIL_BRENT_GLB_D_BRENT' in df.columns:
    df['brent'] = df['COMM_OIL_BRENT_GLB_D_BRENT']  # Sin ffill - datos crudos
    df['brent_z'] = z_score_rolling(df['brent'], ZSCORE_WINDOW)
    df['brent_change_1d'] = pct_change_safe(df['brent'], 12*24)
    df['brent_vol_5d'] = rolling_volatility(df['brent'], 12*24*5)
    df['brent_mom_5d'] = pct_change_safe(df['brent'], 12*24*5)

# --- MACRO: EMBI ---
print("   Calculando features macro (EMBI)...")
if 'CRSK_SPREAD_EMBI_COL_D_EMBI' in df.columns:
    df['embi'] = df['CRSK_SPREAD_EMBI_COL_D_EMBI']  # Sin ffill - datos crudos
    df['embi_z'] = z_score_fixed(df['embi'], **FIXED_NORM_STATS['embi'])
    df['embi_change_5d'] = pct_change_safe(df['embi'], 12*24*5)

# --- MACRO: TASAS ---
print("   Calculando features macro (Tasas)...")
if 'FINC_BOND_YIELD10Y_USA_D_UST10Y' in df.columns:
    df['ust10y'] = df['FINC_BOND_YIELD10Y_USA_D_UST10Y']  # Sin ffill - datos crudos
    # RATE_SPREAD_Z: Rolling z-score for stationarity (fixes distribution shift)
    # Uses col10y - ust10y if col10y available, otherwise 10.0 - ust10y
    # Rolling z-score ensures train and test have similar distributions
    df['rate_spread_z'] = calculate_rate_spread_zscore(df, window=ZSCORE_WINDOW * 12 * 24)
    print(f"   rate_spread_z: rolling z-score, window={ZSCORE_WINDOW * 12 * 24}")

    # EXP-B-001: Rate spread change (carry trade momentum)
    df['rate_spread_change'] = calc_rate_spread_change(df, periods=288)  # 1-day change

if 'FINC_BOND_YIELD2Y_USA_D_DGS2' in df.columns:
    df['ust2y'] = df['FINC_BOND_YIELD2Y_USA_D_DGS2']  # Sin ffill - datos crudos
    # Yield curve slope + z-score normalizado
    if 'ust10y' in df.columns:
        df['curve_slope'] = df['ust10y'] - df['ust2y']
        df['curve_slope_z'] = z_score_rolling(df['curve_slope'], ZSCORE_WINDOW)
        # EXP-B-001: Yield curve z-score (risk-on/off signal)
        df['yield_curve_z'] = calc_yield_curve_zscore(df, window=ZSCORE_WINDOW * 12)

# --- MACRO: CROSS-PAIRS ---
print("   Calculando features cross-pairs...")
if 'FXRT_SPOT_USDMXN_MEX_D_USDMXN' in df.columns:
    df['usdmxn'] = df['FXRT_SPOT_USDMXN_MEX_D_USDMXN']  # Daily data - ffill to 5min
    # NOTE: usdmxn is DAILY frequency, so intraday values are repeated (ffilled)
    # usdmxn_change_1d captures day-over-day change, will be ~0 within same day
    df['usdmxn_change_1d'] = pct_change_safe(df['usdmxn'], 12*24)  # 1 day = 288 bars

if 'FXRT_SPOT_USDCLP_CHL_D_USDCLP' in df.columns:
    df['usdclp'] = df['FXRT_SPOT_USDCLP_CHL_D_USDCLP']  # Daily data - ffill to 5min
    # NOTE: usdclp is DAILY frequency, so intraday values are repeated (ffilled)
    df['usdclp_change_1d'] = pct_change_safe(df['usdclp'], 12*24).clip(-0.05, 0.05)  # 1 day = 288 bars
    df['usdclp_z'] = z_score_rolling(df['usdclp'], ZSCORE_WINDOW)

# --- DS6-DS10: FEATURES ADICIONALES PARA CARRY/COMMODITY/SENTIMENT/FED/FLOWS ---
print("   Calculando features adicionales (DS6-DS10)...")

# =============================================================================
# DS6: CARRY TRADE - DATOS CRUDOS SIN FFILL
# Usar spreads ABSOLUTOS (en puntos porcentuales) y CAMBIOS
# =============================================================================
if 'FINC_BOND_YIELD10Y_COL_D_COL10Y' in df.columns:
    df['col10y'] = df['FINC_BOND_YIELD10Y_COL_D_COL10Y']  # Sin ffill - datos crudos

if 'FINC_BOND_YIELD5Y_COL_D_COL5Y' in df.columns:
    df['col5y'] = df['FINC_BOND_YIELD5Y_COL_D_COL5Y']  # Sin ffill - datos crudos

# SPREAD SOBERANO ABSOLUTO (en puntos porcentuales, ej: 5.5% - 4.2% = 1.3pp)
if 'col10y' in df.columns and 'ust10y' in df.columns:
    df['col10y_ust10y_spread'] = df['col10y'] - df['ust10y']  # Spread absoluto
    df['spread_change_1d'] = df['col10y_ust10y_spread'] - df['col10y_ust10y_spread'].shift(12*24)
    # Z-score con ventana de 20 dias (mas variacion)
    df['spread_z_20d'] = z_score_rolling(df['col10y_ust10y_spread'], 20*12*24)
    # Normalizar spread a rango util (dividir por 10 para que este en ~0.1-0.5)
    df['spread_normalized'] = (df['col10y_ust10y_spread'] / 10.0).clip(-0.5, 0.5)

# PENDIENTE CURVA COLOMBIA (slope absoluto)
if 'col10y' in df.columns and 'col5y' in df.columns:
    df['col_curve_slope'] = df['col10y'] - df['col5y']  # En pp
    df['col_curve_change_1d'] = df['col_curve_slope'] - df['col_curve_slope'].shift(12*24)
    df['col_curve_normalized'] = (df['col_curve_slope'] / 2.0).clip(-1, 1)

# PENDIENTE CURVA USA (slope absoluto)
if 'ust10y' in df.columns and 'ust2y' in df.columns:
    df['usa_curve_slope'] = df['ust10y'] - df['ust2y']  # En pp
    df['usa_curve_change_1d'] = df['usa_curve_slope'] - df['usa_curve_slope'].shift(12*24)
    df['usa_curve_normalized'] = (df['usa_curve_slope'] / 2.0).clip(-1, 1)
    # Indicador de curva invertida (muy importante para recesion)
    df['usa_curve_inverted'] = (df['usa_curve_slope'] < 0).astype(float)

# TPM Colombia y Fed Funds - Usar NIVELES y DIFERENCIAL ABSOLUTO
if 'POLR_POLICY_RATE_COL_M_TPM' in df.columns:
    df['tpm'] = df['POLR_POLICY_RATE_COL_M_TPM']  # Sin ffill - datos crudos
    df['tpm_normalized'] = (df['tpm'] / 15.0).clip(0, 1)  # Normalizado 0-15%

if 'POLR_FED_FUNDS_USA_M_FEDFUNDS' in df.columns:
    df['fedfunds'] = df['POLR_FED_FUNDS_USA_M_FEDFUNDS']  # Sin ffill - datos crudos
    df['fedfunds_normalized'] = (df['fedfunds'] / 10.0).clip(0, 1)  # Normalizado 0-10%

# DIFERENCIAL DE POLITICA (TPM - FEDFUNDS) - Absoluto, no z-score
if 'tpm' in df.columns and 'fedfunds' in df.columns:
    df['policy_spread'] = df['tpm'] - df['fedfunds']  # En pp
    df['policy_spread_normalized'] = (df['policy_spread'] / 10.0).clip(-0.5, 1)
    # Indicadores binarios de regimen de tasas
    df['col_hiking'] = (df['tpm'].diff(12*24*30) > 0).astype(float)  # TPM subiendo
    df['fed_hiking'] = (df['fedfunds'].diff(12*24*30) > 0).astype(float)  # Fed subiendo
    df['carry_favorable'] = (df['policy_spread'] > 2.0).astype(float)  # Spread > 2pp

# IBR overnight - Usar spread vs TPM absoluto
if 'FINC_RATE_IBR_OVERNIGHT_COL_D_IBR' in df.columns:
    df['ibr'] = df['FINC_RATE_IBR_OVERNIGHT_COL_D_IBR']  # Sin ffill - datos crudos
    if 'tpm' in df.columns:
        df['ibr_tpm_spread'] = df['ibr'] - df['tpm']  # Spread absoluto
        df['ibr_tpm_normalized'] = df['ibr_tpm_spread'].clip(-2, 2) / 2.0  # Normalizado
        # Tension interbancaria: IBR muy por encima de TPM
        df['interbank_stress'] = (df['ibr_tpm_spread'] > 0.5).astype(float)

# --- COMMODITY (DS7): WTI, Cafe, Oro, TOT, COLCAP - DATOS CRUDOS SIN FFILL ---
if 'COMM_OIL_WTI_GLB_D_WTI' in df.columns:
    df['wti'] = df['COMM_OIL_WTI_GLB_D_WTI']  # Sin ffill - datos crudos
    df['wti_z'] = z_score_rolling(df['wti'], ZSCORE_WINDOW)
    df['wti_change_1d'] = pct_change_safe(df['wti'], 12*24)
    if 'brent' in df.columns:
        df['brent_wti_spread'] = df['brent'] - df['wti']
        df['brent_wti_spread_z'] = z_score_rolling(df['brent_wti_spread'], ZSCORE_WINDOW)

if 'COMM_AGRI_COFFEE_GLB_D_COFFEE' in df.columns:
    df['coffee'] = df['COMM_AGRI_COFFEE_GLB_D_COFFEE']  # Sin ffill - datos crudos
    df['coffee_z'] = z_score_rolling(df['coffee'], ZSCORE_WINDOW)
    df['coffee_change_1d'] = pct_change_safe(df['coffee'], 12*24)

if 'COMM_METAL_GOLD_GLB_D_GOLD' in df.columns:
    df['gold'] = df['COMM_METAL_GOLD_GLB_D_GOLD']  # Sin ffill - datos crudos
    df['gold_z'] = z_score_rolling(df['gold'], ZSCORE_WINDOW)
    df['gold_change_1d'] = pct_change_safe(df['gold'], 12*24)

if 'FTRD_TERMS_TRADE_COL_M_TOT' in df.columns:
    df['tot'] = df['FTRD_TERMS_TRADE_COL_M_TOT']  # Sin ffill - datos crudos
    df['tot_z'] = z_score_rolling(df['tot'], ZSCORE_WINDOW)
    df['tot_change_1m'] = pct_change_safe(df['tot'], 12*24*30)

if 'EQTY_INDEX_COLCAP_COL_D_COLCAP' in df.columns:
    df['colcap'] = df['EQTY_INDEX_COLCAP_COL_D_COLCAP']  # Sin ffill - datos crudos
    df['colcap_z'] = z_score_rolling(df['colcap'], ZSCORE_WINDOW)
    df['colcap_change_1d'] = pct_change_safe(df['colcap'], 12*24)

# --- RISK SENTIMENT (DS8): VIX percentile, USDMXN z-score - DATOS CRUDOS SIN FFILL ---
if 'vix' in df.columns:
    # VIX percentile de 20 dias (240 bars)
    df['vix_percentile_20d'] = df['vix'].rolling(window=12*24*20, min_periods=50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
    )
    df['vix_change_1d'] = pct_change_safe(df['vix'], 12*24)

if 'embi' in df.columns:
    df['embi_change_1d'] = pct_change_safe(df['embi'], 12*24)
    df['embi_percentile_20d'] = df['embi'].rolling(window=12*24*20, min_periods=50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
    )

if 'usdmxn' in df.columns:
    df['usdmxn_z'] = z_score_rolling(df['usdmxn'], ZSCORE_WINDOW)

# =============================================================================
# DS9: FED WATCH - DATOS CRUDOS SIN FFILL
# Usar INDICADORES BINARIOS DE REGIMEN
# =============================================================================
if 'INFL_CPI_ALL_USA_M_CPIAUCSL' in df.columns:
    df['cpi_usa'] = df['INFL_CPI_ALL_USA_M_CPIAUCSL']  # Sin ffill - datos crudos
    # Cambio mensual (MoM inflation)
    df['cpi_mom'] = pct_change_safe(df['cpi_usa'], 12*24*30)
    # Inflacion alta: MoM > 0.3% (anualizado ~3.6%)
    df['inflation_hot'] = (df['cpi_mom'] > 0.003).astype(float)
    # Inflacion muy alta: MoM > 0.5% (anualizado ~6%)
    df['inflation_crisis'] = (df['cpi_mom'] > 0.005).astype(float)
    # Tendencia inflacionaria
    df['cpi_accelerating'] = (df['cpi_mom'] > df['cpi_mom'].shift(12*24*30)).astype(float)

if 'INFL_PCE_USA_M_PCEPI' in df.columns:
    df['pce_usa'] = df['INFL_PCE_USA_M_PCEPI']  # Sin ffill - datos crudos
    df['pce_mom'] = pct_change_safe(df['pce_usa'], 12*24*30)
    # PCE es preferido por la Fed
    df['pce_above_target'] = (df['pce_mom'] > 0.00167).astype(float)  # >2% anualizado

if 'LABR_UNEMPLOYMENT_USA_M_UNRATE' in df.columns:
    df['unrate'] = df['LABR_UNEMPLOYMENT_USA_M_UNRATE']  # Sin ffill - datos crudos
    # Desempleo bajo (mercado laboral apretado) < 4%
    df['labor_tight'] = (df['unrate'] < 4.0).astype(float)
    # Desempleo alto > 5%
    df['labor_weak'] = (df['unrate'] > 5.0).astype(float)
    # Desempleo subiendo (recesion inminente)
    df['unemployment_rising'] = (df['unrate'].diff(12*24*30) > 0.2).astype(float)

# UST 2Y - El mejor predictor de movimientos de Fed
if 'ust2y' in df.columns:
    df['ust2y_z'] = z_score_rolling(df['ust2y'], ZSCORE_WINDOW)
    df['ust2y_change_1d'] = pct_change_safe(df['ust2y'], 12*24)
    # UST2Y alto (>4%) indica Fed restrictiva
    df['rates_restrictive'] = (df['ust2y'] > 4.0).astype(float)
    # UST2Y bajo (<2%) indica Fed acomodaticia
    df['rates_accommodative'] = (df['ust2y'] < 2.0).astype(float)

if 'ust10y' in df.columns:
    df['ust10y_z'] = z_score_rolling(df['ust10y'], ZSCORE_WINDOW)

# Regimen Fed combinado
if 'inflation_hot' in df.columns and 'labor_tight' in df.columns:
    # Fed hawkish: inflacion alta Y mercado laboral apretado
    df['fed_hawkish'] = ((df['inflation_hot'] == 1) & (df['labor_tight'] == 1)).astype(float)
    # Fed dovish: inflacion baja Y desempleo alto
    df['fed_dovish'] = ((df['inflation_hot'] == 0) & (df.get('labor_weak', 0) == 1)).astype(float)

# =============================================================================
# DS10: FLOWS FUNDAMENTALS - DATOS CRUDOS SIN FFILL
# Usar valores NORMALIZADOS y CAMBIOS
# =============================================================================
if 'RSBP_CURRENT_ACCOUNT_COL_Q_CACCT' in df.columns:
    df['cuenta_corriente'] = df['RSBP_CURRENT_ACCOUNT_COL_Q_CACCT']  # Sin ffill - datos crudos
    # Cuenta corriente normalizada (-10B a 0 -> -1 a 0)
    df['ca_normalized'] = (df['cuenta_corriente'] / 10000).clip(-1, 0)
    # Cuenta corriente mejorando (deficit reduciendose)
    df['ca_improving'] = (df['cuenta_corriente'].diff(26280) > 0).astype(float)

if 'FTRD_EXPORTS_TOTAL_COL_M_EXPUSD' in df.columns:
    df['exports'] = df['FTRD_EXPORTS_TOTAL_COL_M_EXPUSD']  # Sin ffill - datos crudos
    # Tendencia de exportaciones
    df['exports_growing'] = (df['exports'].diff(12*24*30) > 0).astype(float)
    # Exportaciones z-score
    df['exports_z'] = z_score_rolling(df['exports'], ZSCORE_WINDOW)

if 'FTRD_IMPORTS_TOTAL_COL_M_IMPUSD' in df.columns:
    df['imports'] = df['FTRD_IMPORTS_TOTAL_COL_M_IMPUSD']  # Sin ffill - datos crudos
    df['imports_growing'] = (df['imports'].diff(12*24*30) > 0).astype(float)
    df['imports_z'] = z_score_rolling(df['imports'], ZSCORE_WINDOW)

# Trade balance
if 'exports' in df.columns and 'imports' in df.columns:
    df['trade_balance'] = df['exports'] - df['imports']
    df['trade_balance_z'] = z_score_rolling(df['trade_balance'], ZSCORE_WINDOW)
    df['trade_improving'] = (df['trade_balance'].diff(12*24*30) > 0).astype(float)

if 'FXRT_REER_BILATERAL_COL_M_ITCR' in df.columns:
    df['itcr'] = df['FXRT_REER_BILATERAL_COL_M_ITCR']  # Sin ffill - datos crudos
    # ITCR z-score (captura desviacion de la norma)
    df['itcr_z'] = z_score_rolling(df['itcr'], ZSCORE_WINDOW)
    # Desviacion de la media de 1 año (mas corto para mas variacion)
    itcr_1y_mean = df['itcr'].rolling(window=12*24*365, min_periods=500).mean()
    df['itcr_deviation'] = ((df['itcr'] - itcr_1y_mean) / (itcr_1y_mean + 1e-10)).clip(-0.3, 0.3)
    # ITCR cambiando (COP apreciandose o depreciandose)
    df['itcr_change_1m'] = pct_change_safe(df['itcr'], 12*24*30)

if 'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT' in df.columns:
    df['reserves'] = df['RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT']  # Sin ffill - datos crudos
    # Reservas z-score
    df['reserves_z'] = z_score_rolling(df['reserves'], ZSCORE_WINDOW)
    # Reservas cayendo (BanRep defendiendo peso)
    df['reserves_falling'] = (df['reserves'].diff(12*24*30) < 0).astype(float)

# --- TEMPORAL ---
print("   Calculando features temporales...")
df['hour_utc'] = df['datetime'].dt.hour
df['hour_cot'] = df['hour_utc'] - 5  # UTC-5 = Colombia
df['hour_sin'], df['hour_cos'] = encode_cyclical(df['hour_cot'], 24)
df['dow'] = df['datetime'].dt.dayofweek
df['dow_sin'], df['dow_cos'] = encode_cyclical(df['dow'], 7)

# Session liquida (8am - 1pm COT)
df['session_liquid'] = ((df['hour_cot'] >= LIQUID_HOURS[0]) &
                        (df['hour_cot'] < LIQUID_HOURS[1])).astype(float)

print(f"   Total features calculados: {len(df.columns)}")


# =============================================================================
# 3.5. FORWARD-FILL MACRO FEATURES (NO LIMIT)
# =============================================================================
print("\n" + "-" * 80)
print("3.5. FORWARD-FILL MACRO FEATURES (NO LIMIT - propagate until new value)")
print("-" * 80)

# NO LIMIT: Macro data is valid until new data arrives
# - Daily data (DXY, VIX, etc.): propagate through entire day/weekend
# - Monthly data (CPI, TPM, etc.): propagate through entire month
# - NEVER use backfill (backward fill)
MACRO_COLS_TO_FILL = [
    # Daily frequency data
    'embi', 'dxy', 'vix', 'brent', 'ust10y', 'ust2y',
    'usdmxn', 'usdclp', 'wti', 'coffee', 'gold',
    'col10y', 'col5y', 'ibr', 'colcap',
    # Monthly/Quarterly frequency data
    'tpm', 'fedfunds', 'tot', 'cpi_usa', 'pce_usa', 'unrate',
    'cuenta_corriente', 'exports', 'imports',
    'itcr', 'reserves',
    # Derived z-scores and changes (fill remaining NaN after calculation)
    'embi_z', 'dxy_z', 'dxy_mom_5d', 'rate_spread_z',
    'vix_level', 'vix_z', 'brent_z', 'curve_slope',
    'usdmxn_change_1d', 'usdclp_change_1d',  # Daily FX pairs resampled to 5min
]

print("   NaN counts BEFORE forward-fill:")
for col in MACRO_COLS_TO_FILL:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        if nan_count > 0:
            print(f"      {col}: {nan_count:,} ({nan_pct:.1f}%)")

# P0-10 FIX: ffill with limit to prevent stale data propagation
FFILL_LIMIT = 144  # 12 hours max (144 bars * 5 min)
print(f"\n   Applying forward-fill (limit={FFILL_LIMIT} bars = 12 hours)...")
for col in MACRO_COLS_TO_FILL:
    if col in df.columns:
        nans_pre = df[col].isna().sum()
        df[col] = df[col].ffill(limit=FFILL_LIMIT)  # P0-10: Limited ffill
        nans_post = df[col].isna().sum()
        nans_filled = nans_pre - nans_post
        if nans_filled > 0:
            print(f"      {col}: filled {nans_filled:,} NaNs, {nans_post:,} remaining")


# =============================================================================
# 3.6. EXPLICIT NAN STRATEGY (Fill remaining NaN with neutral values)
# =============================================================================
print("\n" + "-" * 80)
print("3.6. EXPLICIT NAN STRATEGY (FILL REMAINING WITH NEUTRAL)")
print("-" * 80)

FILLNA_STRATEGY = {
    'dxy_z': 0.0, 'vix_z': 0.0, 'embi_z': 0.0, 'brent_z': 0.0,
    'rate_spread_z': 0.0, 'dxy_mom_5d': 0.0,
    'dxy_change_1d': 0.0, 'brent_change_1d': 0.0, 'usdmxn_change_1d': 0.0,
    'vix_regime': 1.0, 'vix_level': 1.0,
    # CLOSE-ONLY features (SSOT v2.0.0)
    'volatility_pct': 0.15, 'trend_z': 0.0,  # Neutral values
}

print("   Applying explicit fillna strategy:")
for col, fill_value in FILLNA_STRATEGY.items():
    if col in df.columns:
        nans_before = df[col].isna().sum()
        if nans_before > 0:
            df[col] = df[col].fillna(fill_value)
            print(f"      {col}: filled {nans_before:,} NaNs with {fill_value}")



# =============================================================================
# 5. CREAR LOS 10 DATASETS
# =============================================================================
print("\n" + "-" * 80)
print("4. CREANDO 10 DATASETS")
print("-" * 80)

# P0.1 FIX: Include raw_log_ret_5m in base columns for PnL calculation
base_cols = ['datetime', 'open', 'high', 'low', 'close', 'raw_log_ret_5m']

# -------------------------------------------------------------------------
# DS1: MINIMAL (14 features)
# -------------------------------------------------------------------------
print("\n   DS1_MINIMAL (14 features)")
print("   Filosofia: Baseline para validar pipeline")

ds1_features = [
    # OHLC (4)
    # Retornos (3)
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
    # Tecnicos (3)
    'rsi_9', 'atr_pct', 'bb_position',
    # Macro (2)
    'dxy_z', 'vix_level',
    # Temporal (2)
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS2: TECHNICAL_MTF (18 features)
# -------------------------------------------------------------------------
print("   DS2_TECHNICAL_MTF (18 features)")
print("   Filosofia: Multi-timeframe tecnico puro")

ds2_features = [
    # OHLC (4)
    # Retornos multi-TF (4)
    'log_ret_5m', 'log_ret_15m', 'log_ret_1h', 'log_ret_4h',
    # RSI multi-TF (3)
    'rsi_9', 'rsi_9_15m', 'rsi_9_1h',
    # ATR multi-TF (2)
    'atr_pct', 'atr_pct_1h',
    # Trend (2)
    'adx_14', 'sma_ratio',
    # Volatility (1)
    'bb_position',
    # Temporal (2)
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS3: MACRO_CORE (26 features) - RECOMENDADO (EXP-B-001)
# -------------------------------------------------------------------------
print("   DS3_MACRO_CORE (26 features) [RECOMENDADO - EXP-B-001]")
print("   Filosofia: Balance optimo tecnico + macro para USD/COP")
print("   EXP-B-001: 18 market features + 2 state (TradingEnv) = 20 obs_dim")

ds3_features = [
    # EXP-B-001 - Exact 18 market features (order matters!)
    # Retornos (4) - EXP-B-001: +log_ret_1d
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'log_ret_1d',
    # Tecnicos (4) - CLOSE-ONLY - EXP-B-001: +rsi_21
    'rsi_9', 'rsi_21', 'volatility_pct', 'trend_z',
    # Macro - Dolar (2)
    'dxy_z', 'dxy_change_1d',
    # Macro - Riesgo (2)
    'vix_z', 'embi_z',
    # Macro - Commodities (1)
    'brent_change_1d',
    # Tasas (2) - EXP-B-001: +rate_spread_change
    'rate_spread_z', 'rate_spread_change',
    # Cross-pair (1)
    'usdmxn_change_1d',
    # EXP-B-001: NEW macro features (2)
    'yield_curve_z', 'gold_change_1d',
    # Temporal (2) - for model but not in SSOT observation
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS4: COST_AWARE (18 features)
# -------------------------------------------------------------------------
print("   DS4_COST_AWARE (18 features)")
print("   Filosofia: Filtros para reducir overtrading")

ds4_features = [
    # OHLC (4)
    # Retornos + Lags (4)
    'log_ret_5m', 'log_ret_1h', 'ret_lag_1', 'ret_lag_3',
    # Momentum ajustado (2)
    'ret_atr_adj', 'momentum_6',
    # Filtros de senal (4)
    'rsi_9', 'rsi_extreme', 'adx_14', 'adx_strong',
    # Volatilidad (2)
    'atr_percentile', 'vol_regime',
    # Macro (2)
    'dxy_z', 'vix_z',
    # Temporal (2) - Removido session_liquid (siempre=1)
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS5: REGIME (26 features)
# -------------------------------------------------------------------------
print("   DS5_REGIME (26 features)")
print("   Filosofia: Maximo contexto para detectar cambios de regimen")

ds5_features = [
    # OHLC (4)
    # Retornos multi-TF (4)
    'log_ret_5m', 'log_ret_15m', 'log_ret_1h', 'log_ret_4h',
    # Tecnicos (4)
    'rsi_9', 'atr_pct', 'adx_14', 'bb_position',
    # Macro - Dolar completo (4)
    'dxy_z', 'dxy_change_1d', 'dxy_mom_5d', 'dxy_vol_5d',
    # Macro - Riesgo completo (4)
    'vix_z', 'vix_regime', 'embi_z', 'embi_change_5d',
    # Macro - Commodities (2)
    'brent_change_1d', 'brent_vol_5d',
    # Tasas completo (2)
    'rate_spread', 'curve_slope_z',
    # Cross-pairs (2) - NOTE: both are DAILY data resampled to 5min
    'usdmxn_change_1d', 'usdclp_change_1d',
    # Temporal (3)
    'hour_sin', 'hour_cos', 'dow_sin',
]

# -------------------------------------------------------------------------
# DS6: CARRY_TRADE (18 features) - CORREGIDO con spreads absolutos
# -------------------------------------------------------------------------
print("   DS6_CARRY_TRADE (18 features)")
print("   Filosofia: Diferenciales de tasas ABSOLUTOS para capturar flujos carry trade")

ds6_features = [
    # Retornos (3)
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
    # Spread soberano absoluto COL-USA (3)
    'spread_normalized', 'spread_change_1d', 'spread_z_20d',
    # Curvas de rendimiento (3)
    'col_curve_normalized', 'usa_curve_normalized', 'usa_curve_inverted',
    # Politica monetaria (3)
    'policy_spread_normalized', 'ibr_tpm_normalized', 'carry_favorable',
    # Indicadores binarios de regimen (2)
    'col_hiking', 'fed_hiking',
    # Riesgo (2)
    'embi_z', 'vix_regime',
    # Temporal (2)
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS7: COMMODITY_BASKET (16 features) - CORREGIDO sin tot_z
# -------------------------------------------------------------------------
print("   DS7_COMMODITY_BASKET (16 features)")
print("   Filosofia: Commodities exportados por Colombia (petroleo, cafe, oro)")

ds7_features = [
    # Retornos (2)
    'log_ret_5m', 'log_ret_1h',
    # Petroleo principal (3)
    'brent_z', 'brent_change_1d', 'brent_mom_5d',
    # WTI y spread (2)
    'wti_z', 'brent_wti_spread_z',
    # Otros commodities (4)
    'coffee_z', 'coffee_change_1d', 'gold_z', 'gold_change_1d',
    # Mercado local (2) - Removido tot_z (96% zeros)
    'colcap_z', 'colcap_change_1d',
    # Riesgo (2)
    'vix_z', 'dxy_z',
    # Temporal (2)
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS8: RISK_SENTIMENT (19 features) - CORREGIDO sin usdmxn_ret_5m
# -------------------------------------------------------------------------
print("   DS8_RISK_SENTIMENT (19 features)")
print("   Filosofia: Risk-On/Risk-Off puro para mercados emergentes")

ds8_features = [
    # Retornos (2)
    'log_ret_5m', 'log_ret_1h',
    # Volatilidad global (5)
    'vix_level', 'vix_z', 'vix_regime', 'vix_change_1d', 'vix_percentile_20d',
    # Riesgo pais (4)
    'embi_z', 'embi_change_1d', 'embi_change_5d', 'embi_percentile_20d',
    # Cross-pairs lideres (3) - NOTE: both are DAILY data resampled to 5min
    'usdmxn_change_1d', 'usdmxn_z', 'usdclp_change_1d',
    # Safe haven (2)
    'gold_change_1d', 'dxy_change_1d',
    # Equity local (1)
    'colcap_change_1d',
    # Tecnico (2)
    'rsi_9', 'bb_position',
    # Temporal (2)
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS9: FED_WATCH (17 features) - CORREGIDO con indicadores binarios
# -------------------------------------------------------------------------
print("   DS9_FED_WATCH (17 features)")
print("   Filosofia: Regimen Fed via indicadores binarios, no z-scores")

ds9_features = [
    # Retornos (3)
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
    # Regimen Fed combinado (2)
    'fed_hawkish', 'fed_dovish',
    # Inflacion USA - Binarios (3)
    'inflation_hot', 'inflation_crisis', 'cpi_accelerating',
    # Empleo USA - Binarios (3)
    'labor_tight', 'labor_weak', 'unemployment_rising',
    # Tasas USA - Binarios (2)
    'rates_restrictive', 'rates_accommodative',
    # Curva USA (2)
    'usa_curve_normalized', 'usa_curve_inverted',
    # Temporal (2)
    'hour_sin', 'hour_cos',
]

# -------------------------------------------------------------------------
# DS10: FLOWS_FUNDAMENTALS (14 features) - CORREGIDO v3 sin z-scores
# -------------------------------------------------------------------------
print("   DS10_FLOWS_FUNDAMENTALS (14 features)")
print("   Filosofia: Flujos de capital - solo normalizados y tendencias (sin z-scores)")

ds10_features = [
    # Retornos (2)
    'log_ret_5m', 'log_ret_1h',
    # Flujos de capital - Normalizados (2)
    'ca_normalized', 'ca_improving',
    # Comercio - Solo tendencias (2)
    'exports_growing', 'trade_improving',
    # Competitividad COP (2)
    'itcr_deviation', 'itcr_change_1m',
    # Reservas (1)
    'reserves_falling',
    # Riesgo macro (1)
    'embi_z',
    # Temporal (2)
    'hour_sin', 'hour_cos',
]


def create_dataset(df, features, name):
    """Crea dataset con features especificados"""
    cols = base_cols + [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"      [!] Features faltantes: {missing}")

    df_out = df[cols].copy()

    # Verificar blacklist
    for col in BLACKLIST:
        if col in df_out.columns:
            print(f"      [!] Removiendo blacklisted: {col}")
            df_out = df_out.drop(columns=[col])

    return df_out


df_ds1 = create_dataset(df, ds1_features, 'DS1')
df_ds2 = create_dataset(df, ds2_features, 'DS2')
df_ds3 = create_dataset(df, ds3_features, 'DS3')
df_ds4 = create_dataset(df, ds4_features, 'DS4')
df_ds5 = create_dataset(df, ds5_features, 'DS5')
df_ds6 = create_dataset(df, ds6_features, 'DS6')
df_ds7 = create_dataset(df, ds7_features, 'DS7')
df_ds8 = create_dataset(df, ds8_features, 'DS8')
df_ds9 = create_dataset(df, ds9_features, 'DS9')
df_ds10 = create_dataset(df, ds10_features, 'DS10')


# =============================================================================
# 6. FILTRAR HORARIO Y FECHA
# =============================================================================
print("\n" + "-" * 80)
print("5. FILTRANDO HORARIO Y FECHA")
print("-" * 80)

def filter_market_hours(df, start_hour_utc=13, end_hour_utc=17, end_minute=55):
    """Filtra L-V, 13:00-17:55 UTC (= 8:00-12:55 COT)"""
    df = df.copy()
    hour = df['datetime'].dt.hour
    minute = df['datetime'].dt.minute
    dow = df['datetime'].dt.dayofweek

    mask = (
        (dow < 5) &
        (
            ((hour >= start_hour_utc) & (hour < end_hour_utc)) |
            ((hour == end_hour_utc) & (minute <= end_minute))
        )
    )
    return df[mask].copy()

def apply_date_cutoff(df, start_date):
    """Aplica fecha de corte"""
    return df[df['datetime'] >= start_date].copy()

def drop_warmup_nans(df, required_cols):
    """Elimina filas con NaN en columnas criticas"""
    cols_exist = [c for c in required_cols if c in df.columns]
    return df.dropna(subset=cols_exist, how='any')


# Aplicar filtros a todos los datasets
datasets = {
    'DS1_MINIMAL': (df_ds1, ['log_ret_5m', 'rsi_9', 'dxy_z']),
    'DS2_TECHNICAL_MTF': (df_ds2, ['log_ret_5m', 'rsi_9', 'adx_14']),
    'DS3_MACRO_CORE': (df_ds3, ['log_ret_5m', 'log_ret_1d', 'rsi_21', 'dxy_z', 'vix_z', 'volatility_pct', 'trend_z', 'rate_spread_z', 'rate_spread_change', 'yield_curve_z', 'gold_change_1d']),
    'DS4_COST_AWARE': (df_ds4, ['log_ret_5m', 'rsi_9', 'dxy_z']),
    'DS5_REGIME': (df_ds5, ['log_ret_5m', 'dxy_z', 'vix_z']),
    'DS6_CARRY_TRADE': (df_ds6, ['log_ret_5m', 'spread_normalized', 'policy_spread_normalized']),
    'DS7_COMMODITY_BASKET': (df_ds7, ['log_ret_5m', 'brent_z', 'coffee_z']),
    'DS8_RISK_SENTIMENT': (df_ds8, ['log_ret_5m', 'vix_z', 'embi_z']),
    'DS9_FED_WATCH': (df_ds9, ['log_ret_5m', 'fed_hawkish', 'inflation_hot']),
    'DS10_FLOWS_FUNDAMENTALS': (df_ds10, ['log_ret_5m', 'ca_normalized', 'itcr_z']),
}

filtered_datasets = {}
for name, (df_temp, required) in datasets.items():
    df_filtered = filter_market_hours(df_temp)
    df_filtered = apply_date_cutoff(df_filtered, START_DATE)
    df_filtered = drop_warmup_nans(df_filtered, required)
    df_filtered = df_filtered.reset_index(drop=True)
    filtered_datasets[name] = df_filtered
    print(f"   {name}: {len(df_filtered):,} filas, {len(df_filtered.columns)} cols")


# =============================================================================
# 7. VALIDACION DE CALIDAD
# =============================================================================
print("\n" + "-" * 80)
print("6. VALIDACION DE CALIDAD")
print("-" * 80)

def validate_dataset(df, name):
    """Valida calidad del dataset"""
    issues = []

    # Nulls
    null_counts = df.isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        issues.append(f"Nulls en {len(cols_with_nulls)} columnas")

    # Infinitos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    cols_with_inf = inf_counts[inf_counts > 0]
    if len(cols_with_inf) > 0:
        issues.append(f"Infinitos en {len(cols_with_inf)} columnas")

    # Columnas de zeros
    for col in numeric_cols:
        if col in ['open', 'high', 'low', 'close']:
            continue
        zero_pct = (df[col] == 0).sum() / len(df) * 100
        if zero_pct > 95:
            issues.append(f"{col}: {zero_pct:.1f}% zeros")

    # Blacklist check
    for col in BLACKLIST:
        if col in df.columns:
            issues.append(f"Blacklisted: {col}")

    status = "[OK]" if len(issues) == 0 else "[!]"
    print(f"   {status} {name}")

    if issues:
        for issue in issues[:5]:
            print(f"       - {issue}")
    else:
        print(f"       Fecha: {df['datetime'].min().date()} a {df['datetime'].max().date()}")
        print(f"       Dias: {df['datetime'].dt.date.nunique()}")

    return len(issues) == 0

all_valid = True
for name, df_temp in filtered_datasets.items():
    if not validate_dataset(df_temp, name):
        all_valid = False


# =============================================================================
# 8. GUARDAR DATASETS + NORM_STATS (L2 -> L3 -> L4 Pipeline Integration)
# =============================================================================
print("\n" + "-" * 80)
print("7. GUARDANDO DATASETS + NORM_STATS")
print("-" * 80)

# Base columns that are NOT features (excluded from normalization)
# P0.1 FIX: Include raw_log_ret_5m - it's for PnL calculation, not a model feature
BASE_COLS = ['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'raw_log_ret_5m']

for name, df_temp in filtered_datasets.items():
    # Renombrar datetime a timestamp
    df_save = df_temp.rename(columns={'datetime': 'timestamp'})

    filename = f"RL_{name}.csv"
    filepath = OUTPUT_RL / filename
    df_save.to_csv(filepath, index=False)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"   {filename}: {len(df_save):,} filas, {len(df_save.columns)} cols, {size_mb:.1f} MB")

    # Generate norm_stats.json for this dataset
    # Extract feature columns (exclude base columns)
    feature_cols = [c for c in df_save.columns if c not in BASE_COLS]

    try:
        norm_stats_path = save_norm_stats(
            df=df_save,
            output_dir=OUTPUT_RL,
            dataset_name=name,
            feature_columns=feature_cols
        )
        print(f"      + {norm_stats_path.name}")
    except Exception as e:
        print(f"      [ERROR] Failed to save norm_stats for {name}: {e}")


# =============================================================================
# 9. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN FINAL - 10 DATASETS RL USD/COP")
print("=" * 80)

print(f"""
DATASETS GENERADOS EN: {OUTPUT_RL}

+---------------------------+----------+--------+----------------------+
| Dataset                   | Features | Filas  | Uso                  |
+---------------------------+----------+--------+----------------------+""")

recommendations = {
    'DS1_MINIMAL': 'Validar pipeline',
    'DS2_TECHNICAL_MTF': 'Trend-following',
    'DS3_MACRO_CORE': 'PRODUCCION',
    'DS4_COST_AWARE': 'Anti-overtrading',
    'DS5_REGIME': 'Attention/Transf',
    'DS6_CARRY_TRADE': 'Carry trade flows',
    'DS7_COMMODITY_BASKET': 'Commodity export',
    'DS8_RISK_SENTIMENT': 'Risk-On/Off',
    'DS9_FED_WATCH': 'Fed expectations',
    'DS10_FLOWS_FUNDAMENTALS': 'BOP/Swing trade',
}

for name, df_temp in filtered_datasets.items():
    n_features = len(df_temp.columns) - 5  # Menos OHLC + timestamp
    n_rows = len(df_temp)
    uso = recommendations.get(name, '')
    marker = " ***" if name == 'DS3_MACRO_CORE' else (" **" if name == 'DS6_CARRY_TRADE' else "")
    print(f"| {name:<25} | {n_features:>8} | {n_rows:>6,} | {uso:<20} |{marker}")

print("+---------------------------+----------+--------+----------------------+")

print(f"""
ORDEN DE EXPERIMENTACION RECOMENDADO:

  FASE 1 - CORE:
    1. DS1_MINIMAL         -> Sharpe > 0.3?  -> Continuar
    2. DS3_MACRO_CORE      -> Sharpe > 0.5?  -> USAR EN PRODUCCION ***
    3. DS6_CARRY_TRADE     -> Alto impacto esperado (0.6-1.0) **

  FASE 2 - ESPECIALIZACION:
    4. DS8_RISK_SENTIMENT  -> Para regimenes volatiles
    5. DS7_COMMODITY_BASKET -> Si correlacion Brent-COP es fuerte
    6. DS9_FED_WATCH       -> Ciclos de Fed

  FASE 3 - AVANZADO:
    7. DS4_COST_AWARE      -> Solo si overtrading
    8. DS5_REGIME          -> Solo con arquitecturas attention
    9. DS10_FLOWS_FUNDAMENTALS -> Swing trading, largo plazo

  FASE 4 - ENSEMBLE:
    Combinar DS3 + DS6 + DS8 con votacion mayoritaria o ponderada por VIX regime

COLUMNAS BLACKLISTED (nunca incluidas):
  {', '.join(BLACKLIST)}

NORMALIZACION APLICADA:
  - Retornos: clip [-0.05, 0.05]
  - Z-scores: FIXED (v15) from training period, clip [-4, 4]
  - RSI/ADX: sin normalizar (ya bounded 0-100)
  - BB position: 0-1
  - Percentiles: 0-1

ESTRATEGIA DE DATOS:
  - SIN FFILL: Datos crudos de la fuente
  - NaN donde no hay datos (no se rellenan gaps)
  - Expansion diario->5min ya hecha en consolidado
  - Filtrado de warmup NaN al final con dropna()

FECHA CORTE: {START_DATE} (evita warmup Q1 2020)
""")

# Mostrar sample de DS3 (recomendado)
print("-" * 80)
print("SAMPLE - DS3_MACRO_CORE (ultimas 5 filas):")
print("-" * 80)
print(filtered_datasets['DS3_MACRO_CORE'].tail().to_string())
