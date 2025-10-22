# -*- coding: utf-8 -*-
"""
DAG: usdcop_m5__07_l6_backtest_referencia
=========================================
Layer: L6 - BACKTEST DE REFERENCIA (Hedge-Fund Grade, TEST neto de costos)
Bucket salida: usdcop-l6-backtest

Produce artefactos para dashboard:
- metrics/kpis_test.json, metrics/kpis_val.json
- metrics/kpis_test_rolling.json (60-90d + bandas)
- trades/trade_ledger.parquet
- returns/daily_returns.parquet
- meta/backtest_manifest.json

Entradas:
- L4: usdcop-l4-rlready/{test_df.parquet,val_df.parquet,
         env_spec.json, reward_spec.json, cost_model.json, checks_report.json}
- (Opcional) Bundle L5: usdcop-l5-serving/*/ (policy.onnx, latency.json)

Reglas clave:
- Siempre medir en TEST (y opcional VAL), neto de costos.
- Costos: usar turn_cost_t1 (bps), spread/slippage/fees de L4 si existen.
- Política por defecto si no hay ONNX: baseline flat (posición = 0).
- Trazabilidad completa (hashes, rutas, timestamps).
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import io
import json
import math
import os
import re
import tempfile
import uuid

import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# Import manifest writer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from scripts.write_manifest_example import write_manifest, create_file_metadata
import boto3
from botocore.client import Config

# =========================
# Configuración del DAG
# =========================
DAG_ID = "usdcop_m5__07_l6_backtest_referencia"
BUCKET_L4 = "usdcop-l4-rlready"
BUCKET_L5 = "usdcop-l5-serving"
BUCKET_L6 = "usdcop-l6-backtest"

DEFAULT_ARGS = {
    "owner": "rl-research",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

SCHEDULE = None  # manual

# =========================
# Utilidades
# =========================
def _sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _read_parquet_from_s3(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    obj = hook.get_key(key, bucket_name=bucket)
    data = obj.get()["Body"].read()
    return pd.read_parquet(io.BytesIO(data))

def _read_json_from_s3(hook: S3Hook, bucket: str, key: str) -> Dict[str, Any]:
    obj = hook.get_key(key, bucket_name=bucket)
    data = obj.get()["Body"].read()
    return json.loads(data)

def _write_json_to_s3(hook: S3Hook, bucket: str, key: str, payload: Dict[str, Any]):
    hook.load_string(json.dumps(payload, indent=2, default=str), key=key, bucket_name=bucket, replace=True)

def _write_parquet_to_s3(hook: S3Hook, bucket: str, key: str, df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    hook.load_bytes(buf.getvalue(), key=key, bucket_name=bucket, replace=True)

def _percentile(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))

def _dd_curve(returns: np.ndarray) -> Tuple[float, np.ndarray]:
    # Drawdown de curva de equity acumulada
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    dd = np.where(peak > 1e-12, (equity - peak) / peak, 0.0)
    return float(np.min(dd, initial=0.0)), dd

def _time_to_recover(dd: np.ndarray) -> int:
    # TTR aproximado en pasos (barras); convertiremos a días al consolidar
    ttr = 0
    in_dd = False
    cur_len = 0
    for v in dd:
        if v < 0:
            in_dd = True
            cur_len += 1
        else:
            if in_dd:
                ttr = max(ttr, cur_len)
            in_dd = False
            cur_len = 0
    if in_dd:
        ttr = max(ttr, cur_len)
    return int(ttr)

def _ulcer_index(dd: np.ndarray) -> float:
    # UI = sqrt(mean(D^2)), donde D son drawdowns negativos (en positivo su magnitud)
    if dd.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(np.minimum(dd, 0.0)) * (-1))))

def _sortino(returns: np.ndarray, target: float = 0.0) -> float:
    """
    Calculate annualized Sortino ratio.

    Sortino = (mean_excess / downside_std) * sqrt(252)
    where downside_std is computed only from negative excess returns.
    """
    if returns.size == 0:
        return 0.0
    excess = returns - target
    dr = excess[excess < 0]
    down_std = np.std(dr) if dr.size else 1e-9
    mu = float(np.mean(excess))
    # FIXED: Added sqrt(252) for annualization
    return float(mu / down_std * np.sqrt(252)) if down_std > 1e-9 else 0.0

def _sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (mean_excess / std_excess) * sqrt(252)
    """
    if returns.size == 0:
        return 0.0
    excess = returns - risk_free
    mu, sd = float(np.mean(excess)), float(np.std(excess))
    # FIXED: Added sqrt(252) for annualization
    return float(mu / sd * np.sqrt(252)) if sd > 1e-9 else 0.0

def _calmar(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    # Annualized return / Max DD
    annual_return = _annualize_from_daily(returns)
    max_dd, _ = _dd_curve(returns)
    max_dd = abs(max_dd)
    return float(annual_return / max_dd) if max_dd > 1e-9 else 0.0

def _annualize_from_daily(returns: np.ndarray) -> float:
    # Asumiendo retornos diarios
    periods_per_year = 252
    mu = float(np.mean(returns))
    return float(mu * periods_per_year)

def _cagr_from_daily(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    # CAGR = (1 + total_return)^(252/n_days) - 1
    total_return = np.prod(1 + returns) - 1
    n_days = len(returns)
    if n_days == 0:
        return 0.0
    return float((1.0 + total_return) ** (252 / n_days) - 1.0)

def _value_at_risk_bps(daily: np.ndarray, q: float) -> float:
    # VaR en bps (1bp=1e-4) suponiendo retornos en decimales
    if daily.size == 0:
        return float("nan")
    var = np.percentile(daily, 100 * (1.0 - q))
    return float(var * 1e4)

def _expected_shortfall_bps(daily: np.ndarray, q: float) -> float:
    if daily.size == 0:
        return float("nan")
    thr = np.percentile(daily, 100 * (1.0 - q))
    tail = daily[daily <= thr]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail) * 1e4)

def _group_daily_net_returns(df_episodes: pd.DataFrame) -> pd.DataFrame:
    # df_episodes: columnas ['episode_id','ret_net']
    # Sumamos por episodio (día) => serie diaria neta
    g = (df_episodes.groupby("episode_id")["ret_net"].sum()
         .sort_index())
    return g.reset_index().rename(columns={"ret_net": "ret_net_daily"})

# =========================
# Tareas del DAG
# =========================
def generate_run_id(**ctx):
    run_id = f"L6_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ctx["ti"].xcom_push(key="run_id", value=run_id)
    print(f"Generated run_id: {run_id}")
    return run_id

def discover_inputs(**ctx):
    """
    Localiza en L4 los archivos necesarios y elige último dataset READY.
    Optionally detecta el bundle L5 más reciente con policy.onnx/latency.json.
    """
    s3 = S3Hook(aws_conn_id="minio_conn")

    # Buscar último directorio con fecha en L4
    l4_dirs = s3.list_keys(bucket_name=BUCKET_L4, prefix="date=")
    if not l4_dirs:
        raise ValueError("No se encontraron datos en L4")
    
    # Ordenar y tomar el más reciente
    l4_dirs = [d for d in l4_dirs if d.startswith("date=")]
    l4_dates = sorted(list(set([d.split("/")[0] for d in l4_dirs])))
    if not l4_dates:
        raise ValueError("No se encontraron fechas válidas en L4")
    
    latest_date = l4_dates[-1]
    l4_prefix = f"{latest_date}/"
    
    # Verificar checks_report READY
    checks_key = f"{l4_prefix}checks_report.json"
    if not s3.check_for_key(checks_key, bucket_name=BUCKET_L4):
        raise ValueError(f"L4 checks_report.json no encontrado en {l4_prefix}")
    
    checks = _read_json_from_s3(s3, BUCKET_L4, checks_key)
    if checks.get("status") != "READY":
        raise ValueError(f"L4 no READY, status={checks.get('status')}")

    # Verificar archivos necesarios
    needed = ["test_df.parquet", "env_spec.json", "reward_spec.json", "cost_model.json"]
    missing = [k for k in needed if not s3.check_for_key(f"{l4_prefix}{k}", bucket_name=BUCKET_L4)]
    if missing:
        raise ValueError(f"Faltan archivos en L4: {missing}")

    # Detectar bundle L5 (opcional)
    latest_onnx = None
    latest_latency = None
    
    try:
        l5_dirs = s3.list_keys(bucket_name=BUCKET_L5, prefix="")
        if l5_dirs:
            # Buscar archivos ONNX
            onnx_files = [k for k in l5_dirs if k.endswith((".onnx", "policy.onnx", "model.onnx"))]
            if onnx_files:
                # Tomar el más reciente
                onnx_files.sort()
                latest_onnx = onnx_files[-1]
                
                # Buscar latency.json en el mismo directorio
                onnx_dir = "/".join(latest_onnx.split("/")[:-1])
                lat_key = f"{onnx_dir}/latency.json" if onnx_dir else "latency.json"
                if s3.check_for_key(lat_key, bucket_name=BUCKET_L5):
                    latest_latency = lat_key
    except Exception as e:
        print(f"Warning: No se pudo acceder a L5: {e}")

    # Guardar referencias
    ctx["ti"].xcom_push(key="l4_prefix", value=l4_prefix)
    ctx["ti"].xcom_push(key="l4_checks_key", value=checks_key)
    ctx["ti"].xcom_push(key="l4_test_key", value=f"{l4_prefix}test_df.parquet")
    ctx["ti"].xcom_push(key="l4_val_key", value=f"{l4_prefix}val_df.parquet")
    ctx["ti"].xcom_push(key="l4_env_key", value=f"{l4_prefix}env_spec.json")
    ctx["ti"].xcom_push(key="l4_reward_key", value=f"{l4_prefix}reward_spec.json")
    ctx["ti"].xcom_push(key="l4_cost_key", value=f"{l4_prefix}cost_model.json")
    ctx["ti"].xcom_push(key="l5_onnx_key", value=latest_onnx or "")
    ctx["ti"].xcom_push(key="l5_latency_key", value=latest_latency or "")

    return {
        "l4_ready": True,
        "l4_date": latest_date,
        "onnx_found": bool(latest_onnx),
        "latency_found": bool(latest_latency),
    }

def load_frames_and_specs(split: str, **ctx):
    """
    Carga {split}_df.parquet + specs y devuelve DF + metadatos (hashes, rutas).
    split in {"test", "val"}
    """
    s3 = S3Hook(aws_conn_id="minio_conn")
    run_id = ctx["ti"].xcom_pull(key="run_id")

    key_df = ctx["ti"].xcom_pull(key=f"l4_{split}_key")
    
    # VAL es opcional
    if split == "val" and (not key_df or not s3.check_for_key(key_df, bucket_name=BUCKET_L4)):
        print(f"VAL dataset no encontrado, omitiendo")
        ctx["ti"].xcom_push(key=f"{split}_loaded", value=False)
        return {"loaded": False}

    df = _read_parquet_from_s3(s3, BUCKET_L4, key_df)
    
    # Verificar columnas mínimas
    min_cols = ["episode_id", "t_in_episode", "ret_forward_1"]
    for c in min_cols:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida en {split}_df: {c}")

    # Cargar specs
    env_key = ctx["ti"].xcom_pull(key="l4_env_key")
    rew_key = ctx["ti"].xcom_pull(key="l4_reward_key")
    cost_key = ctx["ti"].xcom_pull(key="l4_cost_key")
    
    env_spec = _read_json_from_s3(s3, BUCKET_L4, env_key) if env_key else {}
    reward_spec = _read_json_from_s3(s3, BUCKET_L4, rew_key) if rew_key else {}
    cost_model = _read_json_from_s3(s3, BUCKET_L4, cost_key) if cost_key else {}

    # Hash del DF para trazabilidad
    parquet_bytes = io.BytesIO()
    df.to_parquet(parquet_bytes, index=False)
    df_hash = _sha256_bytes(parquet_bytes.getvalue())

    meta = {
        "run_id": run_id,
        "split": split,
        "rows": int(len(df)),
        "episodes": int(df["episode_id"].nunique()),
        "source": {
            "bucket": BUCKET_L4,
            "key": key_df,
            "sha256": df_hash,
        },
        "env_spec": env_spec,
        "reward_spec": reward_spec,
        "cost_model": cost_model,
    }

    ctx["ti"].xcom_push(key=f"{split}_df_meta", value=meta)

    # Persistir temporalmente el DF en L6 para otras tareas
    tmp_key = f"{DAG_ID}/_temp/{run_id}/{split}_df.parquet"
    _write_parquet_to_s3(s3, BUCKET_L6, tmp_key, df)
    ctx["ti"].xcom_push(key=f"{split}_tmp_key", value=tmp_key)
    ctx["ti"].xcom_push(key=f"{split}_loaded", value=True)

    print(f"Loaded {split} dataset: {len(df)} rows, {df['episode_id'].nunique()} episodes")
    return {"loaded": True, "meta": meta}

def _load_onnx_session(s3: S3Hook, key_onnx: str):
    """
    Devuelve (session, input_name) o (None, None) si no disponible.
    """
    if not key_onnx:
        return None, None
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime no instalado, usando política baseline")
        return None, None

    try:
        obj = s3.get_key(key_onnx, bucket_name=BUCKET_L5)
        onnx_bytes = obj.get()["Body"].read()
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp.write(onnx_bytes)
            tmp_path = tmp.name
        
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(tmp_path, so)
        
        input_name = sess.get_inputs()[0].name
        os.unlink(tmp_path)
        
        return sess, input_name
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        return None, None

def _infer_actions_onnx(df: pd.DataFrame, obs_cols: List[str], sess, input_name) -> np.ndarray:
    """
    Inferencia ONNX determinística (argmax logits) => acciones en {-1,0,1}.
    Si falla, devuelve array de ceros (flat).
    """
    try:
        X = df[obs_cols].astype(np.float32).values
        
        # Inference por bloques para memoria
        batch_size = 4096
        all_logits = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            outputs = sess.run(None, {input_name: batch})
            # Asumimos que la primera salida son los logits
            logits = outputs[0]
            all_logits.append(logits)
        
        logits = np.vstack(all_logits)
        actions = np.argmax(logits, axis=-1)
        
        # Mapear a {-1, 0, 1} según la política
        n_actions = logits.shape[1]
        if n_actions == 3:
            # 0: short(-1), 1: flat(0), 2: long(1)
            action_map = {0: -1, 1: 0, 2: 1}
            return np.array([action_map[int(a)] for a in actions], dtype=np.int8)
        elif n_actions == 2:
            # 0: flat(0), 1: long(1)
            return actions.astype(np.int8)
        else:
            # Fallback
            return np.zeros(len(df), dtype=np.int8)
            
    except Exception as e:
        print(f"Error en inferencia ONNX: {e}")
        return np.zeros(len(df), dtype=np.int8)

def _baseline_policy(df: pd.DataFrame) -> np.ndarray:
    """Política de referencia si no hay ONNX: flat (0) en todas las barras"""
    return np.zeros(len(df), dtype=np.int8)

def run_backtest_for_split(split: str, **ctx):
    """
    Ejecuta el backtest para TEST (y VAL si existe).
    - Construye posición por barra desde acciones
    - Aplica costos de giro cuando cambia la posición
    - Calcula recompensa por barra: pos(t)*ret_forward_1 - cost_turn_decimal
    - Emite ledger (un registro por trade) y serie diaria
    """
    s3 = S3Hook(aws_conn_id="minio_conn")
    run_id = ctx["ti"].xcom_pull(key="run_id")
    loaded = ctx["ti"].xcom_pull(key=f"{split}_loaded")
    
    if not loaded:
        print(f"Split {split} no cargado, omitiendo backtest")
        return {"split": split, "skipped": True}

    tmp_key = ctx["ti"].xcom_pull(key=f"{split}_tmp_key")
    df = _read_parquet_from_s3(s3, BUCKET_L6, tmp_key)
    
    # Obtener metadatos
    meta = ctx["ti"].xcom_pull(key=f"{split}_df_meta")
    env_spec = meta.get("env_spec", {})
    
    # Detectar columnas de observación
    obs_cols = [c for c in df.columns if re.match(r"^obs_\d{2}(_z_raw)?$", c)]
    if not obs_cols:
        obs_cols = env_spec.get("observation_columns", [])
    if not obs_cols:
        raise ValueError("No se encontraron columnas de observación")
    
    print(f"Usando {len(obs_cols)} columnas de observación: {obs_cols[:3]}...")

    # Intentar cargar ONNX
    onnx_key = ctx["ti"].xcom_pull(key="l5_onnx_key") or ""
    sess, input_name = _load_onnx_session(s3, onnx_key) if onnx_key else (None, None)
    
    if sess is not None:
        print(f"Usando política ONNX desde {onnx_key}")
        actions = _infer_actions_onnx(df, obs_cols, sess, input_name)
        policy = "onnx"
    else:
        print("Usando política baseline (flat)")
        actions = _baseline_policy(df)
        policy = "baseline_flat"

    # Construcción de posición
    position = actions.astype(np.int8)

    # Costos de giro
    if "turn_cost_t1" in df.columns:
        turn_bps = df["turn_cost_t1"].astype(float)
    else:
        # Derivar de componentes
        spread = df.get("spread_proxy_bps_t1", pd.Series(0.0, index=df.index)).astype(float)
        slip = df.get("slip_t1", pd.Series(0.0, index=df.index)).astype(float)
        fee = df.get("fee_bps_t1", pd.Series(0.0, index=df.index)).astype(float)
        turn_bps = spread/2.0 + slip + fee
    
    turn_dec = (turn_bps / 1e4).values  # a decimal

    # Retorno forward
    ret_fwd = df["ret_forward_1"].astype(float).fillna(0.0).values

    # Calcular cambios de posición
    pos_prev = np.roll(position, 1)
    pos_prev[0] = 0
    changed = (position != pos_prev).astype(np.int8)
    
    # Costo por cambio
    cost = turn_dec * changed

    # Retorno neto
    ret_net = position * ret_fwd - cost
    
    # Agregar a dataframe
    df["position"] = position
    df["pos_prev"] = pos_prev
    df["changed"] = changed
    df["turn_cost_dec"] = turn_dec
    df["ret_net"] = ret_net

    # ===== Trade ledger =====
    trades = df.loc[df["changed"] == 1].copy()
    
    if len(trades) > 0:
        # Precio de entrada aproximado
        px_entry = df.get("open_t1", df.get("mid_t1", pd.Series(np.nan, index=df.index))).astype(float)
        trades["px"] = px_entry.loc[trades.index].values
        trades["side"] = trades["position"].map({-1: "short", 0: "flat", 1: "long"})
        trades["qty"] = 1.0  # notional
        
        # Construir ledger con segmentos
        ledger_rows = []
        idx_changes = trades.index.tolist()
        idx_changes.append(len(df))  # sentinel
        
        for i in range(len(idx_changes)-1):
            start = idx_changes[i]
            end = idx_changes[i+1] if i+1 < len(idx_changes) else len(df)
            
            side = int(df.loc[start, "position"])
            if side == 0:
                continue  # Skip flat positions
            
            seg = df.iloc[start:end].copy()
            seg_ret = float(np.sum(seg["ret_net"].values))
            first_cost_bps = float(turn_bps.iloc[start]) if start < len(turn_bps) else 0.0
            
            ledger_rows.append({
                "episode_id": int(seg.iloc[0]["episode_id"]),
                "t_entry": int(seg.iloc[0]["t_in_episode"]),
                "t_exit": int(seg.iloc[-1]["t_in_episode"]) if len(seg) > 0 else int(seg.iloc[0]["t_in_episode"]),
                "side": "long" if side > 0 else "short",
                "px_entry": float(px_entry.iloc[start]) if start < len(px_entry) and not pd.isna(px_entry.iloc[start]) else None,
                "px_exit": float(px_entry.iloc[min(end-1, len(px_entry)-1)]) if end-1 < len(px_entry) and not pd.isna(px_entry.iloc[min(end-1, len(px_entry)-1)]) else None,
                "ret_net_seg": seg_ret,
                "first_turn_cost_bps": first_cost_bps,
                "bars": int(len(seg))
            })
        
        trade_ledger = pd.DataFrame(ledger_rows)
    else:
        trade_ledger = pd.DataFrame()

    # ===== Serie diaria (por episodio) =====
    daily = _group_daily_net_returns(df[["episode_id", "ret_net"]].copy())

    # Persistir outputs
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    base = f"date={date_str}/run_id={run_id}/split={split}"
    
    _write_parquet_to_s3(s3, BUCKET_L6, f"{base}/trades/trade_ledger.parquet", trade_ledger)
    _write_parquet_to_s3(s3, BUCKET_L6, f"{base}/returns/daily_returns.parquet", daily)
    _write_parquet_to_s3(s3, BUCKET_L6, f"{base}/bars_with_positions.parquet", df)

    # Guardar manifiesto
    manifest = {
        "split": split,
        "policy": policy,
        "obs_cols": obs_cols,
        "n_rows": int(len(df)),
        "n_trades": int(len(trade_ledger)),
        "n_days": int(len(daily)),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "df_key": ctx["ti"].xcom_pull(key=f"l4_{split}_key"),
            "env_spec": ctx["ti"].xcom_pull(key="l4_env_key"),
            "reward_spec": ctx["ti"].xcom_pull(key="l4_reward_key"),
            "cost_model": ctx["ti"].xcom_pull(key="l4_cost_key"),
            "onnx_key": ctx["ti"].xcom_pull(key="l5_onnx_key"),
        }
    }
    _write_json_to_s3(s3, BUCKET_L6, f"{base}/meta/backtest_manifest.json", manifest)

    # Guardar claves para métricas
    ctx["ti"].xcom_push(key=f"{split}_daily_key", value=f"{base}/returns/daily_returns.parquet")
    ctx["ti"].xcom_push(key=f"{split}_bars_key", value=f"{base}/bars_with_positions.parquet")
    ctx["ti"].xcom_push(key=f"{split}_ledger_key", value=f"{base}/trades/trade_ledger.parquet")
    ctx["ti"].xcom_push(key=f"{split}_base", value=base)

    print(f"Backtest {split} completado: {len(daily)} días, {len(trade_ledger)} trades")
    return {
        "split": split,
        "policy": policy,
        "days": int(len(daily)),
        "trades": int(len(trade_ledger))
    }

def compute_metrics_for_split(split: str, **ctx):
    """
    Calcula el set de métricas Hedge-Fund Grade para {split}.
    """
    s3 = S3Hook(aws_conn_id="minio_conn")
    run_id = ctx["ti"].xcom_pull(key="run_id")
    
    loaded = ctx["ti"].xcom_pull(key=f"{split}_loaded")
    if not loaded:
        print(f"Split {split} no procesado, omitiendo métricas")
        return {"split": split, "skipped": True}

    # Cargar serie diaria
    daily_key = ctx["ti"].xcom_pull(key=f"{split}_daily_key")
    daily = _read_parquet_from_s3(s3, BUCKET_L6, daily_key)
    daily_vals = daily["ret_net_daily"].astype(float).values

    if len(daily_vals) == 0:
        print(f"No hay retornos diarios para {split}")
        return {"split": split, "no_data": True}

    # Métricas Top Bar
    sortino = _sortino(daily_vals)
    sharpe = _sharpe(daily_vals)
    max_dd, dd_curve = _dd_curve(daily_vals)
    calmar = _calmar(daily_vals)
    vol_ann = float(np.std(daily_vals) * np.sqrt(252))
    cagr = _cagr_from_daily(daily_vals)

    # Colas / drawdowns
    var_99_bps = _value_at_risk_bps(daily_vals, q=0.99)
    es_97_bps = _expected_shortfall_bps(daily_vals, q=0.975)
    peor_dia_bps = float(np.min(daily_vals) * 1e4) if daily_vals.size else float("nan")
    ttr_dias = _time_to_recover(dd_curve)
    ulcer = _ulcer_index(dd_curve)

    # Trading micro: leer ledger
    ledger_key = ctx["ti"].xcom_pull(key=f"{split}_ledger_key")
    ledger = _read_parquet_from_s3(s3, BUCKET_L6, ledger_key)
    
    if len(ledger) > 0:
        wins = ledger["ret_net_seg"] > 0
        win_rate = float(np.mean(wins))
        avg_win = float(np.mean(ledger.loc[wins, "ret_net_seg"])) if wins.any() else 0.0
        avg_loss = float(np.mean(ledger.loc[~wins, "ret_net_seg"])) if (~wins).any() else 0.0
        payoff = float(abs(avg_win / (avg_loss or 1e-9))) if avg_loss != 0 else float("inf")
        
        total_wins = ledger.loc[wins, "ret_net_seg"].sum() if wins.any() else 0.0
        total_losses = abs(ledger.loc[~wins, "ret_net_seg"].sum()) if (~wins).any() else 1e-12
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else float("inf")
        
        expectancy_bps = float(np.mean(ledger["ret_net_seg"]) * 1e4)
        trades_per_day = float(len(ledger) / max(1, len(daily)))
        dur_avg_min = float(np.mean(ledger["bars"])) * 5.0  # cada bar = 5 min
        turnover = trades_per_day * 2  # aproximación
    else:
        win_rate = payoff = profit_factor = expectancy_bps = trades_per_day = dur_avg_min = turnover = 0.0
        avg_win = avg_loss = 0.0

    # Ejecución & costos
    bars_key = ctx["ti"].xcom_pull(key=f"{split}_bars_key")
    bars = _read_parquet_from_s3(s3, BUCKET_L6, bars_key)
    
    slip = bars.get("slip_t1", pd.Series(dtype=float)).astype(float)
    spread = bars.get("spread_proxy_bps_t1", pd.Series(dtype=float)).astype(float)
    fees = bars.get("fee_bps_t1", pd.Series(dtype=float)).astype(float)

    sl_p50 = _percentile(slip.dropna().values, 50) if not slip.empty else float("nan")
    sl_p95 = _percentile(slip.dropna().values, 95) if not slip.empty else float("nan")
    sprd_p50 = _percentile(spread.dropna().values, 50) if not spread.empty else float("nan")
    sprd_p95 = _percentile(spread.dropna().values, 95) if not spread.empty else float("nan")
    fee_bps = float(np.mean(fees.dropna().values)) if not fees.empty else float("nan")

    # Cost-to-Alpha ratio
    cost_bps_mean = float(np.mean(bars["turn_cost_dec"].values) * 1e4)
    alpha_bps_mean = float(np.mean(bars["ret_forward_1"].values) * 1e4)
    cost_to_alpha = float(cost_bps_mean / (abs(alpha_bps_mean) or 1e-9)) if not math.isnan(alpha_bps_mean) else float("nan")

    # Exposición & capacidad
    gross_exposure = float(np.mean(np.abs(bars["position"].values)))
    net_exposure = float(np.mean(bars["position"].values))
    leverage = 1.0  # placeholder
    participation_rate = float("nan")
    beta = float("nan")

    # Latencias
    latency_key = ctx["ti"].xcom_pull(key="l5_latency_key")
    lat_infer_p99 = float("nan")
    lat_e2e_p99 = float("nan")
    
    if latency_key and s3.check_for_key(latency_key, bucket_name=BUCKET_L5):
        try:
            lat = _read_json_from_s3(s3, BUCKET_L5, latency_key)
            lat_infer_p99 = _safe_float(lat.get("pytorch", {}).get("p99", float("nan")))
            lat_e2e_p99 = _safe_float(lat.get("e2e", {}).get("p99", float("nan")))
        except Exception as e:
            print(f"Error leyendo latencias: {e}")

    # Construir KPIs
    kpis = {
        "split": split,
        "top_bar": {
            "CAGR": cagr,
            "Sortino": sortino,
            "Calmar": calmar,
            "Sharpe": sharpe,
            "Vol_annualizada": vol_ann,
            "MaxDD": max_dd
        },
        "colas_y_drawdowns": {
            "VaR_99_bps": var_99_bps,
            "ES_97_5_bps": es_97_bps,
            "peor_dia_bps": peor_dia_bps,
            "time_to_recover_dias": ttr_dias,
            "ulcer_index": ulcer
        },
        "trading_micro": {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff": payoff,
            "profit_factor": profit_factor,
            "expectancy_bps": expectancy_bps,
            "trades_por_dia": trades_per_day,
            "duracion_media_min": dur_avg_min,
            "turnover": turnover
        },
        "ejecucion_costos": {
            "slippage_p50_bps": sl_p50,
            "slippage_p95_bps": sl_p95,
            "spread_p50_bps": sprd_p50,
            "spread_p95_bps": sprd_p95,
            "fees_bps": fee_bps,
            "cost_to_alpha_ratio": cost_to_alpha
        },
        "exposicion_capacidad": {
            "gross_exposure_pct": gross_exposure * 100,
            "net_exposure_pct": net_exposure * 100,
            "leverage": leverage,
            "participation_rate_pct": participation_rate,
            "beta": beta
        },
        "operacion_latencia": {
            "inferencia_p99_ms": lat_infer_p99,
            "end_to_end_p99_ms": lat_e2e_p99
        },
        "meta": {
            "n_dias": int(len(daily)),
            "n_trades": int(len(ledger)),
            "generado": datetime.utcnow().isoformat() + "Z"
        }
    }

    # Guardar KPIs
    base = ctx["ti"].xcom_pull(key=f"{split}_base")
    _write_json_to_s3(s3, BUCKET_L6, f"{base}/metrics/kpis_{split}.json", kpis)

    # Calcular rolling metrics (60/90 días)
    rolling_metrics = []
    windows = [60, 90]
    
    for window in windows:
        if len(daily_vals) >= window:
            for i in range(window, len(daily_vals) + 1):
                window_data = daily_vals[i-window:i]
                
                roll_sortino = _sortino(window_data)
                roll_sharpe = _sharpe(window_data)
                roll_cagr = _cagr_from_daily(window_data)
                
                # Bootstrap para intervalos de confianza
                n_bootstrap = 200
                rng = np.random.default_rng(42)
                sharpe_boots = []
                
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, window, size=window)
                    boot_data = window_data[idx]
                    sharpe_boots.append(_sharpe(boot_data))
                
                sharpe_ci5 = _percentile(np.array(sharpe_boots), 5)
                sharpe_ci95 = _percentile(np.array(sharpe_boots), 95)
                
                rolling_metrics.append({
                    "t": int(i),
                    "window": window,
                    "Sortino": roll_sortino,
                    "Sharpe": roll_sharpe,
                    "CAGR": roll_cagr,
                    "Sharpe_CI5": sharpe_ci5,
                    "Sharpe_CI95": sharpe_ci95
                })
    
    _write_json_to_s3(s3, BUCKET_L6, f"{base}/metrics/kpis_{split}_rolling.json", {"points": rolling_metrics})

    print(f"Métricas calculadas para {split}: Sharpe={sharpe:.3f}, Sortino={sortino:.3f}, MaxDD={max_dd:.3f}")
    return {"split": split, "ok": True}

def finalize_and_publish(**ctx):
    """
    Crea índice consolidado y marca el backtest como READY.
    """
    s3 = S3Hook(aws_conn_id="minio_conn")
    run_id = ctx["ti"].xcom_pull(key="run_id")
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    base = f"date={date_str}/run_id={run_id}"

    # Construir índice
    index = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "bucket": BUCKET_L6,
        "dag_id": DAG_ID,
        "paths": {}
    }

    # Agregar paths para cada split procesado
    for split in ["test", "val"]:
        if ctx["ti"].xcom_pull(key=f"{split}_loaded"):
            split_base = f"{base}/split={split}"
            index["paths"][split] = {
                "kpis": f"{split_base}/metrics/kpis_{split}.json",
                "kpis_rolling": f"{split_base}/metrics/kpis_{split}_rolling.json",
                "trade_ledger": f"{split_base}/trades/trade_ledger.parquet",
                "daily_returns": f"{split_base}/returns/daily_returns.parquet",
                "manifest": f"{split_base}/meta/backtest_manifest.json",
            }

    # Guardar índice
    _write_json_to_s3(s3, BUCKET_L6, f"{base}/index.json", index)
    
    # Marcar como READY
    ready_marker = {
        "ready": True,
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "splits_processed": list(index["paths"].keys())
    }
    _write_json_to_s3(s3, BUCKET_L6, f"{base}/_control/READY", ready_marker)

    print(f"Backtest L6 finalizado y publicado: {run_id}")

    # ========== MANIFEST WRITING ==========
    print("\nWriting manifest for L6 outputs...")

    try:
        # Create boto3 client for MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://minio:9000'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )

        # Create file metadata for all outputs
        files_metadata = []

        # Add files from index
        for split, paths in index.get("paths", {}).items():
            for path_type, path in paths.items():
                try:
                    metadata = create_file_metadata(s3_client, BUCKET_L6, path)
                    files_metadata.append(metadata)
                except Exception as e:
                    print(f"Warning: Could not create metadata for {path}: {e}")

        # Add index and ready marker
        for file_key in [f"{base}/index.json", f"{base}/_control/READY"]:
            try:
                metadata = create_file_metadata(s3_client, BUCKET_L6, file_key)
                files_metadata.append(metadata)
            except Exception as e:
                print(f"Warning: Could not create metadata for {file_key}: {e}")

        # Write manifest
        if files_metadata:
            manifest = write_manifest(
                s3_client=s3_client,
                bucket=BUCKET_L6,
                layer='l6',
                run_id=run_id,
                files=files_metadata,
                status='success',
                metadata={
                    'started_at': datetime.utcnow().isoformat() + 'Z',
                    'pipeline': DAG_ID,
                    'airflow_dag_id': DAG_ID,
                    'splits_processed': list(index["paths"].keys()),
                    'ready': True
                }
            )
            print(f"✅ Manifest written successfully: {len(files_metadata)} files tracked")
        else:
            print("⚠ No files found to include in manifest")

    except Exception as e:
        print(f"❌ Failed to write manifest: {e}")
        # Don't fail the DAG if manifest writing fails
        pass
    # ========== END MANIFEST WRITING ==========

    return {"ready": True, "run_id": run_id, "index_path": f"{base}/index.json"}

# =========================
# Definición del DAG
# =========================
with DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    schedule_interval=SCHEDULE,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["l6", "backtest", "usdcop", "m5", "hedge-fund-grade"],
    description="L6 Backtest de referencia con métricas Hedge-Fund Grade"
) as dag:

    t_run_id = PythonOperator(
        task_id="generate_run_id",
        python_callable=generate_run_id,
    )

    t_discover = PythonOperator(
        task_id="discover_inputs",
        python_callable=discover_inputs,
    )

    t_load_test = PythonOperator(
        task_id="load_test",
        python_callable=load_frames_and_specs,
        op_kwargs={"split": "test"},
    )

    t_load_val = PythonOperator(
        task_id="load_val",
        python_callable=load_frames_and_specs,
        op_kwargs={"split": "val"},
    )

    t_bt_test = PythonOperator(
        task_id="backtest_test",
        python_callable=run_backtest_for_split,
        op_kwargs={"split": "test"},
    )

    t_bt_val = PythonOperator(
        task_id="backtest_val",
        python_callable=run_backtest_for_split,
        op_kwargs={"split": "val"},
    )

    t_metrics_test = PythonOperator(
        task_id="metrics_test",
        python_callable=compute_metrics_for_split,
        op_kwargs={"split": "test"},
    )

    t_metrics_val = PythonOperator(
        task_id="metrics_val",
        python_callable=compute_metrics_for_split,
        op_kwargs={"split": "val"},
    )

    t_finalize = PythonOperator(
        task_id="finalize_and_publish",
        python_callable=finalize_and_publish,
    )

    # Definir dependencias
    t_run_id >> t_discover
    t_discover >> [t_load_test, t_load_val]
    t_load_test >> t_bt_test >> t_metrics_test
    t_load_val >> t_bt_val >> t_metrics_val
    [t_metrics_test, t_metrics_val] >> t_finalize