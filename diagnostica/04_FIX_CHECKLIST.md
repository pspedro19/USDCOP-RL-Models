# USDCOP Trading System - Fix Checklist
## Fecha: 2026-01-08

---

## P0 - CRÍTICO (HOY)

### [ ] 1. Fix Threshold Mismatch (0.30 → 0.10)

**Archivos a modificar:**

1. **Base de datos** - Ejecutar SQL:
   ```bash
   docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading -f /diagnostica/03_P0_FIXES.sql
   ```

2. **Código Python** - `airflow/dags/l5_multi_model_inference.py`:
   ```python
   # Línea ~47, cambiar:
   @dataclass
   class ModelConfig:
       threshold_long: float = 0.10   # ERA 0.30
       threshold_short: float = -0.10  # ERA -0.30
   ```

3. **Verificar** - Ejecutar:
   ```sql
   SELECT model_id, threshold_long FROM config.models;
   -- Debe mostrar 0.10
   ```

### [ ] 2. Implementar StateTracker Persistence

**Archivo:** `src/core/state/state_tracker.py`

```python
# Reemplazar el método _persist_state (actualmente vacío):

def _persist_state(self, state: ModelState) -> None:
    """Persist state to database."""
    import psycopg2

    conn_string = os.getenv('DATABASE_URL', 'postgresql://admin:admin123@usdcop-postgres-timescale:5432/usdcop_trading')

    with psycopg2.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute('''
                INSERT INTO trading_state (model_id, position, entry_price, equity, realized_pnl, trade_count, wins, losses, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (model_id) DO UPDATE SET
                    position = EXCLUDED.position,
                    entry_price = EXCLUDED.entry_price,
                    equity = EXCLUDED.equity,
                    realized_pnl = EXCLUDED.realized_pnl,
                    trade_count = EXCLUDED.trade_count,
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    last_updated = NOW()
            ''', (
                state.model_id,
                state.position,
                state.entry_price,
                state.equity,
                state.realized_pnl,
                state.trade_count,
                state.wins,
                state.losses
            ))
            conn.commit()
```

---

## P1 - ALTO (Esta semana)

### [ ] 3. Fix Macro Data Scraper

**Diagnóstico:**
```bash
# Ver logs del scraper
docker logs usdcop-airflow-worker 2>&1 | grep -i "macro\|banrep" | tail -50
```

**Posibles fixes:**
- Timeout del scraper Selenium
- Cambio en estructura HTML de BanRep
- IP bloqueada (agregar proxy/rotación)

### [ ] 4. Update Drift Monitor to V19 Features

**Archivo:** `airflow/dags/mlops_drift_monitor.py`

```python
# Líneas 56-67, cambiar query de:
query = """
SELECT
    returns_5m, returns_15m, returns_1h,
    volatility_5m, volatility_15m,
    rsi_14, macd, macd_signal,
    ...
"""

# A:
query = """
SELECT
    log_ret_5m, log_ret_1h,
    rsi_9, macd_hist, bb_width,
    vol_ratio, atr_pct,
    hour_sin, hour_cos,
    day_of_week_sin, day_of_week_cos,
    dxy_z, vix_z
FROM dw.fact_features_5m
WHERE timestamp >= NOW() - INTERVAL '30 days'
...
"""
```

---

## P2 - MEDIO (Este mes)

### [ ] 5. Add Realistic Slippage

**Archivo:** `src/trading/paper_trader.py`

```python
# Agregar en execute_signal():

def calculate_slippage(self, price: float, side: str) -> float:
    """Calculate realistic slippage based on market conditions."""
    base_slippage = 0.005  # 0.5% base

    # Add spread (bid-ask)
    spread = 0.003  # 0.3% typical for USDCOP

    # Direction adjustment
    if side == 'LONG':
        return price * (1 + base_slippage + spread/2)
    else:
        return price * (1 - base_slippage - spread/2)
```

### [ ] 6. Review Trading Strategy

**Análisis requerido:**
- Win rate actual: 22.8%
- Analizar trades perdedores
- Revisar features más predictivos
- Considerar retrain con datos 2025-2026

---

## Comandos de Verificación

```bash
# 1. Verificar threshold corregido
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT model_id, threshold_long FROM config.models"

# 2. Verificar trading_state tiene datos
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT * FROM trading_state"

# 3. Verificar macro data reciente
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT fecha, fxrt_index_dxy_usa_d_dxy FROM macro_indicators_daily WHERE fecha > CURRENT_DATE - 7 ORDER BY fecha DESC"

# 4. Restart DAG después de fixes
docker exec usdcop-airflow-webserver airflow dags unpause l5_multi_model_inference
docker exec usdcop-airflow-webserver airflow dags trigger l5_multi_model_inference

# 5. Verificar logs de nueva inferencia
docker logs usdcop-airflow-worker 2>&1 | grep "l5_multi_model" | tail -20
```

---

## Notas Importantes

1. **Antes de aplicar fixes:**
   - Hacer backup de `config.models` tabla
   - Pausar DAG `l5_multi_model_inference`

2. **Después de aplicar fixes:**
   - Reiniciar contenedor `usdcop-airflow-worker`
   - Verificar que inferencias usan nuevo threshold
   - Monitorear P&L en tiempo real

3. **Si algo falla:**
   ```sql
   -- Rollback threshold
   UPDATE config.models SET threshold_long = 0.30, threshold_short = -0.30;
   ```

---

*Checklist generado por auditoría Claude Code - 2026-01-08*
