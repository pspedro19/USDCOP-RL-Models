"""
Seed bi.fact_forecasts with real H=1 predictions from 5 available models.
Trains on all available data, generates predictions for today.

Usage (inside Airflow container):
    python /opt/airflow/scripts/seed_fact_forecasts.py
"""

import sys
import os
import warnings
import uuid

warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/airflow")
os.environ.setdefault("POSTGRES_PASSWORD", "admin123")

import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import StandardScaler

from src.forecasting.models.factory import ModelFactory
from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.contracts import get_horizon_config

print("=" * 60)
print("Seeding bi.fact_forecasts with real H=1 predictions")
print("=" * 60)

# 1. Load OHLCV from DB (bi.dim_daily_usdcop view)
import psycopg2 as _pg

_conn = _pg.connect(
    host="timescaledb", port=5432, database="usdcop_trading",
    user="admin", password="admin123",
)
ohlcv = pd.read_sql(
    "SELECT date, open, high, low, close FROM bi.dim_daily_usdcop ORDER BY date",
    _conn,
)
_conn.close()
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None).dt.normalize()
ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
print(f"Loaded {len(ohlcv)} daily bars from DB")

# 2. Load macro
macro = pd.read_parquet(
    "/opt/airflow/data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"
)
macro = macro.reset_index()
macro.rename(columns={macro.columns[0]: "date"}, inplace=True)
macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None).dt.normalize()

macro_cols = {
    "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
    "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
}
macro_sub = macro[["date"] + list(macro_cols.keys())].copy()
macro_sub.rename(columns=macro_cols, inplace=True)
macro_sub = macro_sub.sort_values("date")
macro_sub["dxy_close_lag1"] = macro_sub["dxy_close_lag1"].shift(1)
macro_sub["oil_close_lag1"] = macro_sub["oil_close_lag1"].shift(1)

df = pd.merge_asof(
    ohlcv.sort_values("date"),
    macro_sub.sort_values("date"),
    on="date",
    direction="backward",
)

# 3. Build features (same as vol_target_backtest.py)
df["return_1d"] = df["close"].pct_change(1)
df["return_5d"] = df["close"].pct_change(5)
df["return_10d"] = df["close"].pct_change(10)
df["return_20d"] = df["close"].pct_change(20)
df["volatility_5d"] = df["return_1d"].rolling(5).std()
df["volatility_10d"] = df["return_1d"].rolling(10).std()
df["volatility_20d"] = df["return_1d"].rolling(20).std()

delta = df["close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi_14d"] = 100 - (100 / (1 + rs))

df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
df["month"] = pd.to_datetime(df["date"]).dt.month
df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)
df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
df["oil_close_lag1"] = df["oil_close_lag1"].ffill()

# Target
df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

feature_cols = list(FEATURE_COLUMNS)
valid = df[feature_cols].notna().all(axis=1) & df["target_return_1d"].notna()
df_valid = df[valid].reset_index(drop=True)
print(
    f"Training data: {len(df_valid)} rows, "
    f"{df_valid['date'].iloc[0].date()} to {df_valid['date'].iloc[-1].date()}"
)

# 4. Train on ALL data except last row
X_train = df_valid[feature_cols].values[:-1].astype(np.float64)
y_train = df_valid["target_return_1d"].values[:-1].astype(np.float64)
X_pred = df_valid[feature_cols].values[-1:].astype(np.float64)

base_price = float(df_valid["close"].iloc[-1])
inference_date = date.today()
target_date = inference_date

print(f"Base price (latest close): {base_price:.2f}")
print(f"Inference date (seed): {inference_date}")

# 5. Train models and predict
models_to_train = ["ridge", "bayesian_ridge", "ard", "xgboost_pure", "hybrid_xgboost"]
horizon_config = get_horizon_config(1)
linear_models = {"ridge", "bayesian_ridge", "ard"}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_pred_scaled = scaler.transform(X_pred)

predictions = []
for model_id in models_to_train:
    try:
        if model_id in linear_models:
            params = None
        else:
            params = horizon_config

        model = ModelFactory.create(model_id, params=params, horizon=1)

        if model.requires_scaling:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_pred_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_pred)

        predicted_return = float(pred[0])
        predicted_price = base_price * np.exp(predicted_return)
        direction = "UP" if predicted_return > 0 else "DOWN"
        signal = 1 if predicted_return > 0 else -1

        predictions.append(
            {
                "model_id": model_id,
                "predicted_return": predicted_return,
                "predicted_return_pct": predicted_return * 100,
                "predicted_price": predicted_price,
                "price_change": predicted_price - base_price,
                "direction": direction,
                "signal": signal,
                "confidence": min(abs(predicted_return) / 0.01, 1.0),
            }
        )

        print(
            f"  {model_id}: ret={predicted_return:+.6f}, "
            f"dir={direction}, price={predicted_price:.2f}"
        )

    except Exception as e:
        print(f"  {model_id}: FAILED - {e}")

print(f"\nSuccessful predictions: {len(predictions)}/{len(models_to_train)}")

# 6. Insert into bi.fact_forecasts
import psycopg2

conn = psycopg2.connect(
    host="timescaledb",
    port=5432,
    database="usdcop_trading",
    user="admin",
    password="admin123",
)
cur = conn.cursor()

iso = inference_date.isocalendar()
inference_week = iso[1]
inference_year = iso[0]

inserted = 0
for p in predictions:
    cur.execute(
        """
        INSERT INTO bi.fact_forecasts (
            id, inference_date, inference_week, inference_year,
            target_date, model_id, horizon_id,
            base_price, predicted_price, predicted_return_pct,
            price_change, direction, signal, confidence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            str(uuid.uuid4()),
            inference_date,
            inference_week,
            inference_year,
            target_date,
            p["model_id"],
            1,
            base_price,
            p["predicted_price"],
            p["predicted_return_pct"],
            p["price_change"],
            p["direction"],
            p["signal"],
            p["confidence"],
        ),
    )
    inserted += 1

conn.commit()
print(f"\nInserted {inserted} rows into bi.fact_forecasts")

# 7. Verify
cur.execute(
    """
    SELECT model_id, predicted_return_pct, direction, signal, confidence
    FROM bi.fact_forecasts
    WHERE horizon_id = 1 AND inference_date = %s
    ORDER BY model_id
    """,
    (inference_date,),
)
rows = cur.fetchall()
print(f"\nVerification ({len(rows)} rows for {inference_date}):")
for r in rows:
    print(
        f"  {r[0]:20s} ret={float(r[1]):+.4f}%  dir={r[2]:4s}  "
        f"sig={r[3]:+d}  conf={float(r[4]):.3f}"
    )

returns = [float(r[1]) for r in rows]
top3 = sorted(returns, key=abs, reverse=True)[:3]
ensemble_ret = np.mean(top3)
ensemble_dir = "UP" if ensemble_ret > 0 else "DOWN"
print(
    f"\n  Ensemble (top-3 by |return|): {ensemble_ret:+.4f}%, "
    f"direction={ensemble_dir}"
)

conn.close()
print("\n" + "=" * 60)
print("Seed COMPLETE")
print("=" * 60)
