"""
L1 - Feature Engineering DAG (CALCULA DESDE TABLAS BASE)
=========================================================
Layer 1: CALCULA features ML desde datos base en PostgreSQL.

ARQUITECTURA (sin features_ml.csv):
  core.usdcop_historical (OHLCV) ─┐
                                  ├──► CALCULAR ──► core.features_ml
  core.macro_indicators (macro)  ─┘

FLUJO:
1. Lee datos BASE desde PostgreSQL (usdcop_historical + macro_indicators)
2. Aplica publication lags a variables macro (evita data leakage)
3. CALCULA indicadores técnicos (RSI, ATR, MACD, Bollinger, ADX)
4. CALCULA z-scores, spreads, regímenes
5. Aplica forward fill donde corresponde
6. Guarda features CALCULADAS en core.features_ml

ESTE DAG ES CRÍTICO:
- Transforma datos base en features para ML
- Ejecutado por bootstrap_system después de cargar datos base
- Ejecutado semanalmente después de L0 data update

Publication Lags (para evitar data leakage):
- exports: 45 days (DANE publica ~45 días después)
- reserves: 5 days (BanRep actualiza semanalmente)
- itcr: 15 days (requiere datos de múltiples países)
- terms_trade: 60 days (publicación trimestral con revisión)
- fdi_inflow: 90 days (trimestral con retraso significativo)

Schedule: Lunes 7:45 AM COT - Después de L0, antes de inferencia
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import os

# Import business days filter
from utils.dag_common import filter_business_days

# =============================================================================
# CONFIGURATION
# =============================================================================

# Publication lags for macro variables (days)
PUBLICATION_LAGS = {
    'exports': 45,
    'imports': 45,
    'reserves': 5,
    'itcr': 15,
    'terms_trade': 60,
    'fdi_inflow': 90,
    'current_account': 90,
}

# =============================================================================
# INDICATOR CODE MAPPING (long codes from historical data -> short codes)
# =============================================================================
# Maps long indicator codes from load_historical_data.py to short codes
# used in feature engineering. Both formats may exist in the database.
INDICATOR_CODE_MAPPING = {
    # Commodities
    'COMM_AGRI_COFFEE_GLB_D_COFFEE': 'coffee',
    'COMM_METAL_GOLD_GLB_D_GOLD': 'gold',
    'COMM_OIL_BRENT_GLB_D_BRENT': 'brent',
    'COMM_OIL_WTI_GLB_D_WTI': 'wti',
    # Country Risk
    'CRSK_SPREAD_EMBI_COL_D_EMBI': 'embi',
    'CRSK_SENTIMENT_CCI_COL_M_CCI': 'cci',
    'CRSK_SENTIMENT_ICI_COL_M_ICI': 'ici',
    # Equity
    'EQTY_INDEX_COLCAP_COL_D_COLCAP': 'colcap',
    # Bonds
    'FINC_BOND_YIELD10Y_COL_D_COL10Y': 'col10y',
    'FINC_BOND_YIELD10Y_USA_D_UST10Y': 'ust10y',
    'FINC_BOND_YIELD2Y_USA_D_DGS2': 'ust2y',
    'FINC_BOND_YIELD5Y_COL_D_COL5Y': 'col5y',
    'FINC_RATE_IBR_OVERNIGHT_COL_D_IBR': 'ibr',
    # Trade
    'FTRD_EXPORTS_TOTAL_COL_M_EXPUSD': 'exports',
    'FTRD_IMPORTS_TOTAL_COL_M_IMPUSD': 'imports',
    'FTRD_TERMS_TRADE_COL_M_TOT': 'terms_trade',
    # FX
    'FXRT_INDEX_DXY_USA_D_DXY': 'dxy',
    'FXRT_REER_BILATERAL_COL_M_ITCR': 'itcr',
    'FXRT_SPOT_USDCLP_CHL_D_USDCLP': 'usdclp',
    'FXRT_SPOT_USDMXN_MEX_D_USDMXN': 'usdmxn',
    # GDP
    'GDPP_REAL_GDP_USA_Q_GDP_Q': 'gdp_usa',
    # Inflation
    'INFL_CPI_ALL_USA_M_CPIAUCSL': 'cpi_usa',
    'INFL_CPI_CORE_USA_M_CPILFESL': 'core_cpi',
    'INFL_CPI_TOTAL_COL_M_IPCCOL': 'cpi_col',
    'INFL_PCE_USA_M_PCEPI': 'pce',
    # Labor
    'LABR_UNEMPLOYMENT_USA_M_UNRATE': 'unemployment',
    # Money Supply
    'MNYS_M2_SUPPLY_USA_M_M2SL': 'm2',
    # Policy Rates
    'POLR_FED_FUNDS_USA_M_FEDFUNDS': 'fedfunds',
    'POLR_POLICY_RATE_COL_D_TPM': 'tpm',
    'POLR_POLICY_RATE_COL_M_TPM': 'tpm',
    'POLR_PRIME_RATE_USA_D_PRIME': 'prime',
    # Production
    'PROD_INDUSTRIAL_USA_M_INDPRO': 'indpro',
    # Balance of Payments
    'RSBP_CURRENT_ACCOUNT_COL_Q_CACCT_Q': 'current_account',
    'RSBP_FDI_INFLOW_COL_Q_FDIIN_Q': 'fdi_inflow',
    'RSBP_FDI_OUTFLOW_COL_Q_FDIOUT_Q': 'fdi_outflow',
    'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT': 'reserves',
    # Sentiment
    'SENT_CONSUMER_USA_M_UMCSENT': 'consumer_sentiment',
    # Volatility
    'VOLT_VIX_USA_D_VIX': 'vix',
}

# Technical indicator parameters
RSI_PERIOD = 9
ATR_PERIOD = 10
ADX_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 10
BB_STD = 2.0
MACD_FAST = 5
MACD_SLOW = 13
MACD_SIGNAL = 5
ZSCORE_WINDOW = 50

# VIX regime thresholds
VIX_THRESHOLDS = [20, 25, 30]

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'pipeline_db'),
        user=os.getenv('POSTGRES_USER', 'pipeline'),
        password=os.getenv('POSTGRES_PASSWORD', 'pipeline_secret')
    )


# =============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# =============================================================================

def calc_log_return(series, periods=1):
    """Log returns with clipping for stability."""
    ret = np.log(series / series.shift(periods))
    return ret.clip(-0.2, 0.2)


def calc_rsi(close, period=14):
    """RSI - Relative Strength Index (0-100)."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_atr(high, low, close, period=14):
    """ATR - Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_atr_pct(high, low, close, period=14):
    """ATR as percentage of price."""
    atr = calc_atr(high, low, close, period)
    return (atr / close) * 100


def calc_adx(high, low, close, period=14):
    """ADX - Average Directional Index (0-100)."""
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
    return dx.rolling(window=period).mean()


def calc_bollinger_position(close, period=20, std=2):
    """Position in Bollinger Bands (0-1)."""
    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    position = (close - lower) / (upper - lower + 1e-10)
    return position.clip(0, 1)


def calc_macd_histogram(close, fast=12, slow=26, signal=9):
    """MACD Histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def calc_sma_ratio(close, period=20):
    """Ratio close / SMA - 1."""
    sma = close.rolling(window=period).mean()
    return (close / sma) - 1


def z_score_rolling(series, window=50, clip_val=4.0):
    """
    Rolling Z-score with clipping - PREVENTS DATA LEAKAGE.

    CRITICAL: This function uses ROLLING window to calculate mean/std,
    ensuring that at time t, only data from [t-window, t] is used.
    This prevents "look-ahead bias" where future information leaks into past.

    WRONG approach (causes leakage):
        z = (x - x.mean()) / x.std()  # Uses ENTIRE dataset mean/std

    CORRECT approach (this function):
        z = (x - rolling_mean) / rolling_std  # Uses only past data

    Args:
        series: Input time series
        window: Rolling window size (default 50 days)
        clip_val: Maximum absolute z-score value (default 4.0)

    Returns:
        Rolling z-score series, clipped to [-clip_val, clip_val]
    """
    mean = series.rolling(window=window, min_periods=10).mean()
    std = series.rolling(window=window, min_periods=10).std()
    z = (series - mean) / (std + 1e-10)
    return z.clip(-clip_val, clip_val)


def pct_change_safe(series, periods=1):
    """Safe percentage change with clipping."""
    return series.pct_change(periods).clip(-0.2, 0.2)


def encode_cyclical(value, max_value):
    """Cyclical encoding (sin/cos)."""
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_raw_data(**context):
    """Load raw OHLCV and macro data from PostgreSQL."""
    print("Loading raw data from PostgreSQL...")

    conn = get_db_connection()

    # Load USD/COP OHLCV
    query_ohlcv = """
        SELECT date, open_price as open, high_price as high,
               low_price as low, close_price as close
        FROM core.usdcop_historical
        ORDER BY date
    """
    df_ohlcv = pd.read_sql(query_ohlcv, conn)
    df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'])

    # Filter out weekends and Colombian holidays
    df_ohlcv = filter_business_days(df_ohlcv, 'date')

    print(f"  OHLCV (business days only): {len(df_ohlcv)} records")

    # Load macro indicators (pivot from long to wide format)
    query_macro = """
        SELECT date, indicator_code, value
        FROM core.macro_indicators
        ORDER BY date, indicator_code
    """
    df_macro_long = pd.read_sql(query_macro, conn)

    if len(df_macro_long) > 0:
        df_macro_long['date'] = pd.to_datetime(df_macro_long['date'])
        df_macro = df_macro_long.pivot(index='date', columns='indicator_code', values='value')
        df_macro = df_macro.reset_index()

        # Apply indicator code mapping (long codes -> short codes)
        # This handles historical data with long codes like 'CRSK_SPREAD_EMBI_COL_D_EMBI'
        # and converts them to short codes like 'embi'
        # Note: Multiple long codes can map to the same short code (e.g., daily and monthly TPM)

        # Group mappings by target short code to handle conflicts
        short_to_long = {}
        for long_code, short_code in INDICATOR_CODE_MAPPING.items():
            if long_code in df_macro.columns:
                if short_code not in short_to_long:
                    short_to_long[short_code] = []
                short_to_long[short_code].append(long_code)

        merged_count = 0
        for short_code, long_codes in short_to_long.items():
            # Start with existing short code data if present, else first long code
            if short_code in df_macro.columns:
                combined = df_macro[short_code].copy()
            else:
                combined = df_macro[long_codes[0]].copy()
                long_codes = long_codes[1:]  # Remove first since we used it

            # Merge in remaining long codes, preferring newer data
            for long_code in long_codes:
                combined = combined.combine_first(df_macro[long_code])

            # Drop all the long codes
            for long_code in short_to_long[short_code]:
                if long_code in df_macro.columns:
                    df_macro = df_macro.drop(columns=[long_code])

            # Set the combined data
            df_macro[short_code] = combined
            merged_count += 1

        if merged_count > 0:
            print(f"  Applied {merged_count} indicator code mappings")

        print(f"  Macro: {len(df_macro)} records, {len(df_macro.columns)-1} indicators")
    else:
        df_macro = pd.DataFrame({'date': []})
        print("  Macro: No data")

    conn.close()

    # Store in XCom
    context['ti'].xcom_push(key='df_ohlcv', value=df_ohlcv.to_json(date_format='iso'))
    context['ti'].xcom_push(key='df_macro', value=df_macro.to_json(date_format='iso'))

    return f"Loaded OHLCV: {len(df_ohlcv)}, Macro: {len(df_macro)}"


def apply_publication_lags(**context):
    """Apply publication lags to macro variables to avoid data leakage."""
    print("Applying publication lags...")

    ti = context['ti']
    df_macro = pd.read_json(ti.xcom_pull(key='df_macro'))
    df_macro['date'] = pd.to_datetime(df_macro['date'])

    for col, lag_days in PUBLICATION_LAGS.items():
        if col in df_macro.columns:
            print(f"  {col}: shift({lag_days} days)")
            df_macro[col] = df_macro[col].shift(lag_days)

    ti.xcom_push(key='df_macro_lagged', value=df_macro.to_json(date_format='iso'))
    return f"Applied {len(PUBLICATION_LAGS)} lags"


def merge_and_ffill(**context):
    """Merge OHLCV with macro data and apply forward fill."""
    print("Merging data and applying ffill...")

    ti = context['ti']
    df_ohlcv = pd.read_json(ti.xcom_pull(key='df_ohlcv'))
    df_macro = pd.read_json(ti.xcom_pull(key='df_macro_lagged'))

    df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'])
    df_macro['date'] = pd.to_datetime(df_macro['date'])

    # Left join to keep only trading days
    df = pd.merge(
        df_ohlcv.sort_values('date'),
        df_macro.sort_values('date'),
        on='date',
        how='left'
    )

    # Forward fill macro data (they don't change daily)
    macro_cols = [c for c in df.columns if c not in ['date', 'open', 'high', 'low', 'close']]
    for col in macro_cols:
        df[col] = df[col].ffill()

    print(f"  Merged: {len(df)} records, {len(df.columns)} columns")

    ti.xcom_push(key='df_merged', value=df.to_json(date_format='iso'))
    return f"Merged: {len(df)} records"


def calculate_features(**context):
    """Calculate all ML features."""
    print("Calculating ML features...")

    ti = context['ti']
    df = pd.read_json(ti.xcom_pull(key='df_merged'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # === RETURNS ===
    print("  Calculating returns...")
    df['log_ret_1d'] = calc_log_return(df['close'], 1)
    df['log_ret_5d'] = calc_log_return(df['close'], 5)
    df['log_ret_22d'] = calc_log_return(df['close'], 22)
    df['ret_lag_1'] = df['log_ret_1d'].shift(1)
    df['ret_lag_5'] = df['log_ret_1d'].shift(5)
    df['momentum_5d'] = df['log_ret_1d'].rolling(5).sum()
    df['momentum_22d'] = df['log_ret_1d'].rolling(22).sum()

    # === TECHNICAL INDICATORS ===
    print("  Calculating technical indicators...")
    df['rsi_9'] = calc_rsi(df['close'], RSI_PERIOD)
    df['rsi_14'] = calc_rsi(df['close'], 14)
    df['atr'] = calc_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
    df['atr_pct'] = calc_atr_pct(df['high'], df['low'], df['close'], ATR_PERIOD)
    df['adx_14'] = calc_adx(df['high'], df['low'], df['close'], ADX_PERIOD)
    df['bb_position'] = calc_bollinger_position(df['close'], BB_PERIOD, BB_STD)
    df['sma_ratio'] = calc_sma_ratio(df['close'], SMA_PERIOD)
    df['macd_hist'] = calc_macd_histogram(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    # RSI extremes and ADX strong
    df['rsi_extreme'] = ((df['rsi_9'] < 25) | (df['rsi_9'] > 75)).astype(float)
    df['adx_strong'] = (df['adx_14'] > 25).astype(float)

    # ATR percentile and volatility regime
    def atr_percentile_rank(x):
        if len(x) < 2:
            return 0.5
        return (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
    df['atr_percentile'] = df['atr'].rolling(ZSCORE_WINDOW).apply(atr_percentile_rank, raw=False)
    df['vol_regime'] = pd.cut(df['atr_percentile'], bins=[-0.01, 0.25, 0.75, 1.01], labels=[0, 1, 2]).astype(float)

    # Return adjusted by ATR
    df['ret_atr_adj'] = (df['log_ret_1d'] / (df['atr_pct'] / 100 + 1e-10)).clip(-3, 3)

    # === MACRO FEATURES ===
    print("  Calculating macro features...")

    # DXY
    if 'dxy' in df.columns:
        df['dxy_z'] = z_score_rolling(df['dxy'], ZSCORE_WINDOW)
        df['dxy_change_1d'] = pct_change_safe(df['dxy'], 1)
        df['dxy_change_5d'] = pct_change_safe(df['dxy'], 5)

    # VIX
    if 'vix' in df.columns:
        df['vix_z'] = z_score_rolling(df['vix'], ZSCORE_WINDOW)
        df['vix_regime'] = pd.cut(
            df['vix'],
            bins=[0, VIX_THRESHOLDS[0], VIX_THRESHOLDS[1], VIX_THRESHOLDS[2], 100],
            labels=[0, 1, 2, 3]
        ).astype(float).ffill()

    # Commodities
    for commodity in ['brent', 'wti', 'gold', 'coffee']:
        if commodity in df.columns:
            df[f'{commodity}_z'] = z_score_rolling(df[commodity], ZSCORE_WINDOW)
            df[f'{commodity}_change_5d'] = pct_change_safe(df[commodity], 5)

    if 'brent' in df.columns:
        df['brent_change_1d'] = pct_change_safe(df['brent'], 1)
        df['brent_vol_22d'] = df['brent'].pct_change().rolling(22).std()

    # EMBI
    if 'embi' in df.columns:
        df['embi_z'] = z_score_rolling(df['embi'], ZSCORE_WINDOW)
        df['embi_change_5d'] = pct_change_safe(df['embi'], 5)

    # Yield curve
    if 'ust10y' in df.columns:
        df['ust10y_z'] = z_score_rolling(df['ust10y'], ZSCORE_WINDOW)

    if 'ust2y' in df.columns and 'ust10y' in df.columns:
        df['curve_slope'] = df['ust10y'] - df['ust2y']
        df['curve_slope_z'] = z_score_rolling(df['curve_slope'], ZSCORE_WINDOW)
        df['curve_inverted'] = (df['curve_slope'] < 0).astype(float)

    # Colombia rates
    if 'col10y' in df.columns and 'ust10y' in df.columns:
        df['col_us_spread'] = df['col10y'] - df['ust10y']
        df['col_us_spread_z'] = z_score_rolling(df['col_us_spread'], ZSCORE_WINDOW)

    # Policy rates
    if 'tpm' in df.columns and 'fedfunds' in df.columns:
        df['policy_spread'] = df['tpm'] - df['fedfunds']
        df['policy_spread_z'] = z_score_rolling(df['policy_spread'], ZSCORE_WINDOW)
        df['carry_favorable'] = (df['policy_spread'] > 2.0).astype(float)

    # Trade balance
    if 'exports' in df.columns and 'imports' in df.columns:
        df['trade_balance'] = df['exports'] - df['imports']
        df['trade_balance_z'] = z_score_rolling(df['trade_balance'], ZSCORE_WINDOW)

    # Cross pairs
    for pair in ['usdmxn', 'usdclp']:
        if pair in df.columns:
            df[f'{pair}_ret_1d'] = pct_change_safe(df[pair], 1)
            df[f'{pair}_z'] = z_score_rolling(df[pair], ZSCORE_WINDOW)

    # COLCAP
    if 'colcap' in df.columns:
        df['colcap_ret_1d'] = pct_change_safe(df['colcap'], 1)
        df['colcap_z'] = z_score_rolling(df['colcap'], ZSCORE_WINDOW)

    # === TEMPORAL FEATURES ===
    print("  Calculating temporal features...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['dow_sin'], df['dow_cos'] = encode_cyclical(df['day_of_week'], 5)
    df['month_sin'], df['month_cos'] = encode_cyclical(df['month'], 12)

    # === TARGET VARIABLE (7 horizontes) ===
    # HORIZONS = [1, 5, 10, 15, 20, 25, 30] - Igual que training pipeline
    print("  Calculating targets for 7 horizons [1, 5, 10, 15, 20, 25, 30]...")
    df['target_1d'] = calc_log_return(df['close'].shift(-1), 1)
    df['target_5d'] = calc_log_return(df['close'].shift(-5), 5)
    df['target_10d'] = calc_log_return(df['close'].shift(-10), 10)
    df['target_15d'] = calc_log_return(df['close'].shift(-15), 15)
    df['target_20d'] = calc_log_return(df['close'].shift(-20), 20)
    df['target_25d'] = calc_log_return(df['close'].shift(-25), 25)
    df['target_30d'] = calc_log_return(df['close'].shift(-30), 30)

    # Drop warmup period
    df = df[df['date'] >= '2020-03-01'].copy()

    print(f"  Final: {len(df)} records, {len(df.columns)} columns")

    ti.xcom_push(key='df_features', value=df.to_json(date_format='iso'))
    return f"Features: {len(df.columns)} columns"


def save_to_database(**context):
    """Save features to PostgreSQL."""
    print("Saving features to PostgreSQL...")

    ti = context['ti']
    df = pd.read_json(ti.xcom_pull(key='df_features'))
    df['date'] = pd.to_datetime(df['date'])

    conn = get_db_connection()
    cursor = conn.cursor()

    # ===========================================================================
    # CONTRATO DE DATOS: core.features_ml
    # ===========================================================================
    # BASE (5): date, close, open, high, low
    # FEATURES (29): Las 29 variables que entran a entrenamiento
    # TARGETS (7): target_1d, target_5d, target_10d, target_15d, target_20d, target_25d, target_30d
    # TOTAL: 5 + 29 + 7 = 41 columnas + source = 42
    # ===========================================================================

    columns_to_save = [
        # === BASE (5) ===
        'date', 'close', 'open', 'high', 'low',

        # === TECHNICAL INDICATORS (16) - Critical for boosting models ===
        # Returns and momentum (4)
        'ret_lag_1', 'ret_lag_5', 'momentum_5d', 'momentum_22d',
        # RSI (3)
        'rsi_9', 'rsi_14', 'rsi_extreme',
        # ATR and volatility (4)
        'atr_pct', 'atr_percentile', 'vol_regime', 'ret_atr_adj',
        # Trend indicators (3)
        'adx_14', 'adx_strong', 'bb_position',
        # MACD and SMA (2)
        'macd_hist', 'sma_ratio',

        # === MACRO/FUNDAMENTAL FEATURES (29) ===
        # DXY (3)
        'dxy', 'dxy_z', 'dxy_change_1d',
        # VIX (3)
        'vix', 'vix_z', 'vix_regime',
        # Commodities - Petróleo (4)
        'brent', 'brent_z', 'wti', 'wti_z',
        # EMBI (2)
        'embi', 'embi_z',
        # Curva de tasas (2)
        'curve_slope', 'curve_slope_z',
        # Spreads Colombia (5)
        'col_us_spread', 'col_us_spread_z',
        'policy_spread', 'policy_spread_z', 'carry_favorable',
        # Commodities adicionales (2)
        'coffee_z', 'gold_z',
        # Macro Colombia (6)
        'terms_trade', 'trade_balance_z',
        'exports', 'fdi_inflow', 'reserves', 'itcr',
        # Cross-currency (2)
        'usdmxn_ret_1d', 'usdclp_ret_1d',

        # === 7 TARGETS (horizontes) ===
        'target_1d', 'target_5d', 'target_10d', 'target_15d',
        'target_20d', 'target_25d', 'target_30d'
    ]

    # Map column names: CSV -> DB
    column_mapping = {
        'close': 'close_price',
        'open': 'open_price',
        'high': 'high_price',
        'low': 'low_price',
    }

    # Filter to existing columns
    cols_exist = [c for c in columns_to_save if c in df.columns]

    # Truncate and reload
    cursor.execute("TRUNCATE TABLE core.features_ml RESTART IDENTITY CASCADE;")

    # Build insert
    db_cols = [column_mapping.get(c, c) for c in cols_exist]
    db_cols.append('source')

    placeholders = ', '.join(['%s'] * len(db_cols))
    insert_sql = f"""
        INSERT INTO core.features_ml ({', '.join(db_cols)})
        VALUES ({placeholders})
    """

    records = []
    for _, row in df.iterrows():
        record = []
        for col in cols_exist:
            val = row[col]
            if col == 'date':
                record.append(row['date'].date() if hasattr(row['date'], 'date') else row['date'])
            elif pd.isna(val):
                record.append(None)
            else:
                record.append(float(val) if isinstance(val, (int, float, np.number)) else val)
        record.append('l1_feature_engineering')
        records.append(tuple(record))

    cursor.executemany(insert_sql, records)
    conn.commit()
    conn.close()

    print(f"  Saved {len(records)} records")
    return f"Saved {len(records)} records"


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

DAG_ID = "03_l1_feature_engineering"

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Layer 1: Feature Engineering Pipeline',
    schedule_interval='45 12 * * 1',  # Lunes 7:45 AM COT - Después de L0, antes de inferencia
    catchup=False,
    tags=['layer1', 'features', 'ml', 'weekly'],
)

# Tasks
t1 = PythonOperator(
    task_id='load_raw_data',
    python_callable=load_raw_data,
    provide_context=True,
    dag=dag,
)

t2 = PythonOperator(
    task_id='apply_publication_lags',
    python_callable=apply_publication_lags,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='merge_and_ffill',
    python_callable=merge_and_ffill,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='calculate_features',
    python_callable=calculate_features,
    provide_context=True,
    dag=dag,
)

t5 = PythonOperator(
    task_id='save_to_database',
    python_callable=save_to_database,
    provide_context=True,
    dag=dag,
)

# Dependencies
t1 >> t2 >> t3 >> t4 >> t5
