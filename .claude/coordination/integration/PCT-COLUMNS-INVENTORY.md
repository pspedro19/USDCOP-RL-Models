---
kind: audit
status: CHARACTERIZED
version: 1.0.0
last_verified: 2026-07-29
scope: read-only
related: BL-42 (.claude/specs/planes/backlog/BL-42-unidades-decimales-signal-normalizada.md)
code_anchors:
  - scripts/pipeline/train_and_export_smart_simple.py
  - airflow/dags/forecast_h5_l7_multiday_executor.py
  - airflow/dags/forecast_h5_l6_weekly_monitor.py
  - airflow/dags/forecast_h5_l5_vol_targeting.py
  - usdcop-trading-dashboard/app/api/production/live/route.ts
---

# Inventario de columnas `_pct` que almacenan DECIMALES — tablas `forecast_h5_*`

**Alcance**: SÓLO LECTURA. Este documento **caracteriza**; no propone ni ejecuta migración.
Ningún `UPDATE`/`ALTER`/`INSERT`/`DELETE` fue emitido contra la base.

**Base auditada**: `usdcop-postgres-timescale` → `usdcop_trading`, 2026-07-29.
**Antecedente**: BL-42 ya declara la regla; este inventario es la evidencia por columna
que faltaba para decidir la fase 2.

---

## 0. Resumen ejecutivo

| | Conteo |
|---|---|
| Columnas `_pct` en `forecast_h5_*` | **17** |
| **Confirmadas MAL** (decimal bajo sufijo `_pct`, con prueba cruzada) | **10** |
| Correctas (puntos porcentuales genuinos) | **5** |
| **No determinables por dato** (columna 100 % NULL) | **2** |
| Filas afectadas (suma de celdas no nulas mal escaladas) | **62** en 4 tablas |
| Llegan a pantalla | **7** de las 10 (vía `/api/production/live` → `/production`) |
| Columnas con **más de un escritor** | **9** de las 10 |
| Columnas con **escritores en desacuerdo de unidad entre sí** | **2** (`cumulative_pnl_pct`, `running_max_dd_pct`) |

**Las tablas están casi vacías** (8–18 filas) y **congeladas desde 2026-07-07** (§6). El defecto
es real pero su exposición hoy es pequeña; eso *baja* la urgencia de migrar el histórico y *sube*
la de arreglar el productor antes de que las tablas se llenen.

---

## 1. Criterio de decisión (declarado antes de mirar los datos)

Una columna `_pct` está **mal** si su valor almacenado es la fracción decimal (0.0146) en vez del
punto porcentual (1.46). La heurística de magnitud sola (`|mediana| < 0.5`) **no basta** — una
columna de retornos semanales genuinamente pequeños la dispararía. Por eso cada veredicto de esta
tabla se apoya en **al menos una prueba cruzada** de las siguientes, en orden de fuerza:

- **P1 — Identidad de precios en la misma fila**: `dirección · (exit_price/entry_price − 1) · leverage`
  reconstruye el retorno verdadero desde columnas independientes del campo sospechoso.
- **P2 — Identidad de agregación en la misma tabla**: una columna acumulada que es exactamente
  `100 ×` la suma de la columna semanal demuestra que **una de las dos** miente sobre su unidad.
- **P3 — Coincidencia con el SSOT congelado**: `config/execution/smart_simple_v1.yaml:64` declara
  `hard_stop_max_pct: 0.03  # 3% maximo`. Un valor 0.03 en DB *es* ese 3 % escrito como decimal.
- **P4 — El código del productor**: la expresión que calcula el valor es una fracción sin `×100`,
  o divide explícitamente por 100 antes de escribir.
- **P5 — El código del consumidor**: el lector multiplica por 100 al leer, con comentario que lo
  declara. Un lector que compensa es una confesión de la unidad real.

**Columna de control**: `forecast_h5_predictions.predicted_return_pct` satisface P1 en puntos
porcentuales (1.6481 ≈ 100·(3420.99/3365.07−1) = 1.6617) y su productor escribe `pred_val * 100`.
Prueba que la convención de puntos porcentuales **existe y se aplica correctamente en algún sitio
del mismo fichero** — las demás no son "otra convención coherente", son una incoherencia.

---

## 2. Veredicto por columna

Con N = 8–10 no se reportan percentiles más finos que la mediana (quant-constitution §6: con N
pequeño se reportan los valores, no estadística decorativa). `n_nn` = filas no nulas.

### 2.1 CONFIRMADAS MAL — decimal bajo sufijo `_pct` (10 columnas, 62 celdas)

| # | Tabla.columna | Tipo SQL | n_tot | n_nn | min | max | avg | mediana(abs) | Prueba |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `forecast_h5_executions.week_pnl_pct` | `double precision` | 8 | 8 | −0.004914 | 0.007898 | 0.002210 | 0.004525 | P1, P2, P4 |
| 2 | `forecast_h5_executions.hard_stop_pct` | `double precision` | 8 | 8 | 0.015522 | 0.030000 | 0.024933 | 0.026927 | P3, P4, P5 |
| 3 | `forecast_h5_executions.take_profit_pct` | `double precision` | 8 | 8 | 0.007761 | 0.015000 | 0.012466 | 0.013464 | P1, P3, P4, P5 |
| 4 | `forecast_h5_signals.hard_stop_pct` | `double precision` | 10 | 10 | 0.015522 | 0.030000 | 0.025946 | 0.028725 | P3, P4, P5 |
| 5 | `forecast_h5_signals.take_profit_pct` | `double precision` | 10 | 10 | 0.007761 | 0.015000 | 0.012973 | 0.014363 | P3, P4, P5 |
| 6 | `forecast_h5_signals.effective_hs_pct` | `double precision` | 10 | **2** | 0.030000 | 0.030000 | 0.030000 | 0.030000 | P3, P4 |
| 7 | `forecast_h5_signals.effective_tp_pct` | `double precision` | 10 | **2** | 0.015000 | 0.015000 | 0.015000 | 0.015000 | P3, P4 |
| 8 | `forecast_h5_subtrades.pnl_pct` | `double precision` | 8 | 8 | −0.004914 | 0.007898 | 0.002210 | 0.004525 | P1, P4 |
| 9 | `forecast_h5_subtrades.pnl_unleveraged_pct` | `double precision` | 8 | 8 | −0.009730 | 0.015000 | 0.003568 | 0.006499 | P1, P4 |
| 10 | `forecast_h5_paper_trading.week_pnl_pct` | `double precision` | 8 | 8 | −0.004914 | 0.007898 | 0.002210 | 0.004525 | P2, P4 |

**El máximo absoluto de las 10 es 0.03.** Ninguna cruza 1 en ninguna fila — la firma exacta de un
decimal, no la de una serie de retornos genuinamente pequeños (que cruzaría 1 alguna vez).

#### Prueba P1 — reconstrucción desde precios (`forecast_h5_executions`, fila `id=10`)

```
signal_date = 2026-05-11   direction = +1   leverage = 0.5265204614731536
entry_price = 3757.21      exit_price = 3813.57      exit_reason = 'take_profit'

retorno crudo   = 3813.57/3757.21 − 1        = 0.015000492…   →  +1.5000 %
retorno apalanc.= 0.015000492 × 0.52652      = 0.007898066…   →  +0.7898 %

ALMACENADO: week_pnl_pct        = 0.007898        (≠ 0.7898  → decimal)
            pnl_unleveraged_pct = 0.015000492…    (≠ 1.5000  → decimal)
            take_profit_pct     = 0.015           (el TP se disparó EXACTAMENTE en +1.5000 %,
                                                   luego 0.015 significa 1.5 %, no 1.5 pp)
            hard_stop_pct       = 0.03            (= 2× TP, coherente con HS multiplier 2.0×)
```

La fila `id=8` (2026-04-27, también `exit_reason='take_profit'`) repite la identidad:
retorno crudo `0.010423859`, `take_profit_pct = 0.010423520`. **El nivel de TP y el retorno
realizado al tocarlo coinciden en la sexta cifra** — sólo posible si ambos están en la misma
escala decimal. Esto es concluyente y no depende de ninguna heurística de magnitud.

#### Prueba P2 — dos unidades **en la misma fila** de `forecast_h5_paper_trading`

`cumulative_pnl_pct` es exactamente `100 ×` la suma corrida de `week_pnl_pct`:

| signal_date | week_pnl_pct | suma corrida | cumulative_pnl_pct | ratio |
|---|---|---|---|---|
| 2026-01-05 | 0.006104 | 0.006104 | **0.6104** | 100× |
| 2026-03-30 | −0.001679 | 0.004425 | **0.4425** | 100× |
| 2026-04-06 | −0.004914 | −0.000489 | **−0.0489** | 100× |
| 2026-04-13 | −0.001665 | −0.002154 | **−0.2154** | 100× |
| 2026-04-27 | 0.005212 | 0.003058 | **0.3058** | 100× |
| 2026-05-04 | 0.002585 | 0.005643 | **0.5643** | 100× |
| 2026-05-11 | 0.007898 | 0.013541 | **1.3541** | 100× |
| 2026-06-22 | 0.004136 | 0.017677 | **1.7677** | 100× |

Dos columnas con el mismo sufijo `_pct`, en la misma fila, difieren en un factor 100. Combinado con
P1 (que fija `week_pnl_pct` como decimal), queda demostrado que **`cumulative_pnl_pct` es la
correcta y `week_pnl_pct` la equivocada** — no al revés.

#### Prueba P3 — el SSOT congelado escribe decimales bajo nombres `_pct`

`config/execution/smart_simple_v1.yaml:63-65`:
```yaml
  hard_stop_min_pct: 0.01     # 1% minimo (mercados tranquilos)
  hard_stop_max_pct: 0.03     # 3% maximo (mercados volatiles)
  tp_ratio: 0.5               # TP = HS * 0.5
```
El comentario del propio config declara que `0.03` significa `3 %`. El máximo observado en las
cuatro columnas de stops es exactamente `0.03` / `0.015`. **El origen de la convención torcida es
el nombre del parámetro en el SSOT**, y se propaga verbatim a la DB.

---

### 2.2 CORRECTAS — puntos porcentuales genuinos (5 columnas)

No tocar. Cualquier migración masiva por sufijo las **rompería**.

| Tabla.columna | n_nn | min | max | Evidencia de que está bien |
|---|---|---|---|---|
| `forecast_h5_predictions.predicted_return_pct` | 18 | −0.9139 | 1.6481 | P1: ≈ `100·(predicted_price/base_price−1)`. Productor escribe `pred_val * 100` |
| `forecast_h5_paper_trading.cumulative_pnl_pct` | 8 | −0.2154 | 1.7677 | P2 (tabla arriba). Productor escribe `cum_pnl * 100` |
| `forecast_h5_paper_trading.running_da_pct` | 8 | 25 | 100 | Directional accuracy; rango 25–100 es inequívocamente pp |
| `forecast_h5_paper_trading.running_da_short_pct` | 8 | 50 | 100 | ídem |
| `forecast_h5_paper_trading.running_max_dd_pct` | 8 | −0.8239 | 0 | Coherente con el pico/valle de `cumulative_pnl_pct` (0.6104 → −0.2154 ⇒ DD ≈ −0.83 pp) |

---

### 2.3 NO DETERMINABLES POR EL DATO — columnas 100 % NULL (2 columnas)

**No puedo demostrar que estén mal: no hay ni un valor que examinar.** Sólo puedo reportar lo que
*escribiría* su productor si llegara a ejecutarse.

| Tabla.columna | n_tot | n_nn | Qué haría el productor | Veredicto |
|---|---|---|---|---|
| `forecast_h5_executions.week_pnl_unleveraged_pct` | 8 | **0** | `forecast_h5_l7_multiday_executor.py:456,490,591` escribe `raw_pnl` (decimal, sin ×100) | **Sospechosa, no probada.** Nunca poblada: el único productor que sí escribió estas 8 filas (`train_and_export_smart_simple.py`) no incluye la columna en su `INSERT` |
| `forecast_h5_paper_trading.running_da_long_pct` | 8 | **0** | `forecast_h5_l6_weekly_monitor.py:202` escribe `round(da_long, 1)` = `long_correct/len×100` → **pp correctos** | **Probablemente correcta**, sin dato que lo confirme |

`forecast_h5_paper_trading.long_pct_8w` también está 100 % NULL; su productor
(`forecast_h5_l6_weekly_monitor.py:183`) escribe `n_long_recent/len × 100` → pp correctos. No es
una columna de retorno, se lista aquí sólo para cerrar el inventario de las 17.

---

## 3. QUIÉN LA ESCRIBE (productores)

Hay **tres familias de escritores** sobre las mismas tablas. Que exista más de una es en sí un
hallazgo: no hay un único camino de escritura, y **no coinciden en las unidades** (§3.4).

### 3.1 Familia A — script de seeding/export (**escribió las filas que hoy existen**)

`scripts/pipeline/train_and_export_smart_simple.py`, función de seeding DB (bloque ~1480-1700).

| Columna escrita | Línea | Expresión | Unidad resultante |
|---|---|---|---|
| `predictions.predicted_return_pct` | **1505** | `round(pred_val * 100, 4)` | ✅ pp |
| `signals.hard_stop_pct` / `take_profit_pct` | **1530** | `wd["hard_stop_pct"]`, `wd["take_profit_pct"]` (del SSOT, decimales) | ❌ decimal |
| `executions.hard_stop_pct` / `take_profit_pct` | **1555-1556**, **1585-1586** | ídem | ❌ decimal |
| `executions.week_pnl_pct` | **1584** | `trade["pnl_pct"] / 100.0` | ❌ decimal |
| `subtrades.pnl_pct` | **1611** | `lev_pnl = trade["pnl_pct"] / 100.0` | ❌ decimal |
| `subtrades.pnl_unleveraged_pct` | **1610** | `raw_pnl = direction*(exit−entry)/entry` | ❌ decimal |
| `paper_trading.week_pnl_pct` | **1633**, 1683 | `pnl_pct = trade["pnl_pct"] / 100.0` | ❌ decimal |
| `paper_trading.cumulative_pnl_pct` | **1683** | `cum_pnl * 100` | ✅ pp |
| `paper_trading.running_max_dd_pct` | **1637** | `dd = (eq−peak)/peak * 100` | ✅ pp |
| `paper_trading.running_da_pct` | **1655** | `n_correct/n_weeks * 100` | ✅ pp |

**Éste es el hallazgo central.** El ledger de entrada `trade["pnl_pct"]` **ya viene en puntos
porcentuales** (lo garantiza `tests/regression/test_return_units.py::test_producer_emits_percentage_points_not_a_decimal`
sobre `_compute_result_metrics`). El mismo bucle, en el mismo fichero, con separación de ~130
líneas, **multiplica por 100 para unas columnas y divide por 100 para otras**:

```python
# línea 1505  — predictions: CORRECTO
round(pred_val * 100, 4)
...
# línea 1584  — executions: DIVIDE por 100 un valor que ya estaba en pp
trade["exit_reason"], trade["pnl_pct"] / 100.0,
...
# línea 1611  — subtrades: idéntico
lev_pnl = trade["pnl_pct"] / 100.0
...
# línea 1633  — paper_trading semanal: idéntico
pnl_pct = trade["pnl_pct"] / 100.0
# línea 1683  — paper_trading acumulado: MULTIPLICA por 100, restaurando pp
wd["final_lev"], pnl_pct, cum_pnl * 100,
```

La división `/100.0` no es un descuido de omisión (olvidar un `×100`): es una **conversión
deliberada y explícita hacia el decimal**, escrita tres veces. Alguien decidió que la DB guarda
decimales — y no actualizó ni el nombre de la columna ni las otras cuatro columnas del mismo
`INSERT`.

**Confirmación de que ESTA familia escribió las filas actuales** (tres evidencias independientes):
1. `config_version = 'smart_simple_v1'` en las 8 filas de `executions` y en `signals` id 4-11 —
   literal hardcodeado en este script (líneas 1516, 1542, 1567). Los DAGs usan `smart_executor_h5_v1`.
2. `executions.week_pnl_unleveraged_pct` es NULL en las 8 filas — esta columna **no aparece** en el
   `INSERT` del script, pero **sí** en el `UPDATE` del DAG L7. Si L7 hubiera cerrado estas
   posiciones, no serían NULL.
3. `paper_trading.gate_status`, `long_pct_8w`, `running_da_long_pct`, `notes` son NULL en las 8
   filas — exactamente las columnas que el script omite y el DAG L6 sí escribe.

### 3.2 Familia B — DAGs de Airflow (ciclo semanal en vivo)

| Fichero:línea | Sentencia | Columnas `_pct` | Expresión / unidad |
|---|---|---|---|
| `airflow/dags/forecast_h5_l5_weekly_signal.py:265` | `INSERT forecast_h5_predictions` | `predicted_return_pct` | del XCom `pred["predicted_return_pct"]` — pp (coherente con la col. de control) |
| `airflow/dags/forecast_h5_l5_weekly_signal.py:292` | `INSERT forecast_h5_signals` | — | no escribe stops en este paso |
| `airflow/dags/forecast_h5_l5_vol_targeting.py:343` | `UPDATE forecast_h5_signals` | `hard_stop_pct`, `take_profit_pct`, `effective_hs_pct`, `effective_tp_pct` | `leverage["hard_stop_pct"] = effective_hs` (línea 308-309), decimales del SSOT → ❌ decimal |
| `airflow/dags/forecast_h5_l7_multiday_executor.py:286` | `INSERT forecast_h5_executions` | `hard_stop_pct`, `take_profit_pct` | del XCom de la señal → ❌ decimal |
| `airflow/dags/forecast_h5_l7_multiday_executor.py:445,453` | `UPDATE` subtrades+executions (hard stop) | `pnl_pct`, `pnl_unleveraged_pct`, `week_pnl_pct`, `week_pnl_unleveraged_pct` (L456) | `raw_pnl = sub_dir*(exit−entry)/entry`; `lev_pnl = raw_pnl*leverage` → ❌ decimal |
| `airflow/dags/forecast_h5_l7_multiday_executor.py:479,487` | ídem (take profit) | ídem (L490) | ❌ decimal |
| `airflow/dags/forecast_h5_l7_multiday_executor.py:571,588` | ídem (week end) | ídem (L591) | `SUM(pnl_pct)` de subtrades → ❌ decimal |
| `airflow/dags/forecast_h5_l7_multiday_executor.py:645,661` | ídem (circuit breaker) | `pnl_pct`, `week_pnl_pct` | ❌ decimal |
| `airflow/dags/forecast_h5_l6_weekly_monitor.py:343` | `INSERT forecast_h5_paper_trading` | `week_pnl_pct`, `cumulative_pnl_pct`, `running_max_dd_pct`, `running_da_*`, `long_pct_8w` | ver §3.4 — **unidades distintas de la Familia A** |

**Confesiones en el propio código de L7** (prueba P4 en su forma más directa):

```python
# forecast_h5_l7_multiday_executor.py:277-282 — usado como multiplicador decimal
tp_price = entry_price * (1 + take_profit_pct)
hs_price = entry_price * (1 - hard_stop_pct)

# :331-332 — y multiplicado por 100 SÓLO para poder imprimirlo como porcentaje
f"TP={tp_price:.2f} ({take_profit_pct*100:.2f}%), "
f"HS={hs_price:.2f} ({hard_stop_pct*100:.2f}%), "
```

Un `_pct` que hay que multiplicar por 100 para escribirlo con el símbolo `%` es, por definición,
no un porcentaje.

**Error de comunicación operativo ya presente en los logs**: las cuatro rutas de salida de L7
formatean el decimal con un `%` pegado **sin** convertir:
```python
# :464, :498, :599, :669
f"[H5-L7] HARD STOP: … pnl={lev_pnl:+.4f}%, …"
```
Una semana de `+0.79 %` se registra en el log del operador como **`pnl=+0.0079%`** — subestimado
100×. No es sólo un problema de esquema: ya está mintiendo en la superficie que un humano lee
durante un incidente.

### 3.3 Familia C — scripts archivados (histórico, no activos)

`scripts/archive/cron_monitor.py:221,229,252,260,285` y `scripts/archive/cron_week_end.py:127,151`
emiten `UPDATE` sobre `forecast_h5_subtrades` / `forecast_h5_executions` con la misma aritmética
decimal. Viven bajo `scripts/archive/` (superseded por los DAGs L6/L7 según
`scripts/archive/README.md`) y no están cableados a ningún DAG ni Makefile. **Se listan por
completitud del inventario de escritores; no son productores vivos.**

### 3.4 Los escritores **no coinciden entre sí** — 2 columnas en disputa

Esto es más grave que el sufijo, porque el valor de la columna depende de **quién la escribió
última**:

| Columna | Familia A (`train_and_export…py`) | Familia B (`…l6_weekly_monitor.py`) | Dato actual |
|---|---|---|---|
| `paper_trading.cumulative_pnl_pct` | `cum_pnl * 100` → **pp** (L1683) | `cumulative += p` sobre decimales → **decimal** (L143-146, 199) | pp (lo escribió A) |
| `paper_trading.running_max_dd_pct` | `(eq−peak)/peak * 100` → **pp** (L1637) | `equity.append(equity[-1] + p)` sobre decimales → **decimal** (L171-178, 204) | pp (lo escribió A) |

**Consecuencia de seguridad — un circuit breaker que no puede dispararse.**
`forecast_h5_l6_weekly_monitor.py:257-260`:
```python
max_dd = -abs(cb.get("max_drawdown_pct", 12.0))      # = -12.0  (puntos porcentuales)
if metrics["running_max_dd_pct"] <= max_dd:          # pero metrics[...] es un DECIMAL
    circuit_breaker = True
```
Un drawdown real del −12 % produce `running_max_dd_pct = −0.12` en L6, que **nunca** es `<= −12.0`.
Con los escritores de la Familia B, **el corta-circuitos por drawdown del monitor semanal está
muerto**. (El mismo umbral en la Familia A sí funciona, porque allí `max_dd` está en pp:
`cb_triggered = consec_losses >= 5 or abs(max_dd) >= 12.0`, línea 1662.)

Esta es la razón principal para tratar el asunto como defecto de producción y no como deuda
cosmética.

### 3.5 Nota de estado: **los escritores de la Familia B están hoy rotos** (causa adyacente)

Las cuatro rutas de `INSERT` de los DAGs usan `ON CONFLICT (signal_date)`:

- `forecast_h5_l5_weekly_signal.py:298` → `ON CONFLICT (signal_date)`
- `forecast_h5_l7_multiday_executor.py:292` → `ON CONFLICT (signal_date)`
- `forecast_h5_l6_weekly_monitor.py:350` → `ON CONFLICT (signal_date)`
- `train_and_export_smart_simple.py:1517, 1543, 1568, 1672` → `ON CONFLICT (signal_date)` (×4)

Pero `database/migrations/064_h5_strategy_id.sql` **eliminó** la unicidad por `signal_date` sola y
la sustituyó por `(signal_date, strategy_id)`. Verificado contra `pg_index`: no existe hoy ningún
índice único sobre `(signal_date)` en `signals`, `executions` ni `paper_trading`. Postgres rechaza
un `ON CONFLICT` que no case con una restricción existente (`42P10`), luego **todos estos `INSERT`
fallan en ejecución**.

Evidencia consistente: las filas más recientes de `executions` son de `created_at = 2026-07-05` con
`updated_at = 2026-07-07`; `signals` no ha crecido desde `2026-07-07 04:48`. Las tablas llevan tres
semanas congeladas.

**Por qué importa para esta decisión**: (a) explica por qué el volumen afectado es de 62 celdas y
no de miles; (b) significa que **arreglar el productor no tiene efecto observable hasta que se
arregle también el `ON CONFLICT`** — y que en el momento en que se arregle, la Familia B empezará a
escribir sus propias unidades (§3.4) sobre las filas de la Familia A. Es un hallazgo separado de
BL-42, pero se cruza con él en el peor momento posible.

---

## 4. QUIÉN LA LEE (consumidores) — ¿llega a una pantalla o a una decisión?

### 4.1 Llega a pantalla — `/production` (**sí**, pero hoy se ve BIEN)

`usdcop-trading-dashboard/app/api/production/live/route.ts` es el único lector de UI. **Ya compensa
la unidad al leer**, con el comentario que lo declara:

| Línea | Código | Efecto |
|---|---|---|
| 66 | `// DB stores HS/TP as fractions (0.03 = 3%), convert to percentages for frontend` | — |
| 74-75 | `hard_stop_pct: hsPct * 100`, `take_profit_pct: tpPct * 100` | corrige señal actual |
| 109 | `// DB stores HS/TP as fractions (0.03 = 3%), convert to percentages` | — |
| 111-113 | `tpPctScaled = tpPct * 100`, `hsPctScaled = hsPct * 100` | corrige posición activa |
| 209-211 | `// DB stores week_pnl_pct as fraction (0.029772 = 2.9772%), convert to %`<br>`const pnlPct = rawPnl * 100;` | corrige PnL, equity curve, Sharpe, PF, maxDD, win rate |
| 254-255 | `hard_stop_pct: tradeHs * 100`, `take_profit_pct: tradeTp * 100` | corrige tabla de trades |
| 323 | `cumulative_pnl_pct: safeNumber(r.cumulative_pnl_pct)` | **sin ×100** — correcto, porque esa columna sí está en pp |

Componentes que renderizan el resultado ya corregido: `components/gm/views/ProductionView.tsx:348,351,424-426`,
`components/production/LivePositionCard.tsx:161,167,311`, `components/production/GuardrailsCard.tsx:59`,
`components/legacy/ProductionLegacy.tsx:911,914`.

**Conclusión de urgencia**: el operador **no** está viendo hoy números 100× equivocados en
`/production`. El lector acertó la unidad real, columna por columna, incluida la excepción de
`cumulative_pnl_pct`. Pero eso significa que **la corrección vive en el frontend y no en un
contrato**, y convierte a este fichero en la trampa principal de la migración (§5).

### 4.2 Llega a una decisión — reloj de PnL / withdrawal protocol (**sí, y con la unidad mal**)

`airflow/dags/control_system_health.py:294-303` alimenta el reloj de PnL que puede disparar
`withdrawal_protocol_triggered`:
```python
SELECT e.week_pnl_pct, p.week_pnl_pct
FROM forecast_h5_executions e JOIN forecast_h5_paper_trading p USING (signal_date)
...
live  = np.array([r[0] for r in rows], dtype=float) / 100.0
paper = np.array([r[1] for r in rows], dtype=float) / 100.0
```
**Este lector divide por 100 otra vez**, asumiendo que la columna está en puntos porcentuales. Está
escrito para la convención *futura* (BL-42 fase 2), no para la actual. Hoy alimenta el reloj con
series 100× más pequeñas de lo que cree.

**Honestidad sobre el impacto**: el estadístico que consume esas series
(`src/monitoring/system_health.py:330-338`) es `max|cumsum(d)| / (σ_d·√k)` — **invariante de
escala**. Multiplicar `live` y `paper` por la misma constante no cambia el z-score ni el veredicto.
Lo mismo aplica al `running_sharpe`. **Por tanto no puedo afirmar que este reloj esté produciendo
hoy un veredicto equivocado; no lo está.** Lo que sí es cierto: el lector y la DB discrepan en la
unidad, y esa discrepancia sólo es inocua mientras el estadístico sea adimensional. Si alguien
añade un umbral absoluto (p. ej. "TE > 50 bps"), pasa a ser un fallo silencioso. Y en la migración
este `/100.0` es una doble conversión más que localizar (§5).

### 4.3 Lector roto — reconciliación (no llega a ninguna parte)

`src/reconciliation/engine.py:185-192` consulta:
```sql
SELECT signal_date, direction, entry_price, exit_price,
       adjusted_leverage, pnl_pct, exit_reason
FROM forecast_h5_executions WHERE signal_date = %s AND status = 'closed'
```
`forecast_h5_executions` **no tiene** ni `adjusted_leverage` ni `pnl_pct` (son `leverage` y
`week_pnl_pct`). Verificado contra la base viva: `ERROR: column "adjusted_leverage" does not exist`.
El motor de reconciliación H5 no puede leer nada. Es un lector muerto — irrelevante para la unidad,
pero relevante para no contarlo como consumidor afectado.

### 4.4 Lectores no afectados

- `usdcop-trading-dashboard/app/api/trading/signals/route.ts:337-345` — lee sólo
  `entry/exit_timestamp`, `direction`, precios y `exit_reason`. Ninguna columna `_pct`.
- `control_system_health.py:234-240` — lee `predictions.predicted_return_pct` (columna correcta)
  para PSI de drift; PSI también es invariante de escala.
- `scripts/ops/backup/feature_data_backup.py:60-64` — vuelca las 5 tablas a parquet sin
  interpretar unidades. **Nota**: los parquet bajo `data/backups/features/` (versionados en git)
  contienen ya los valores decimales; un restore los reintroduce tal cual.

---

## 5. Qué haría falta para arreglarlo (NO EJECUTADO — para decisión de otra persona)

### 5.1 Alcance mínimo

1. **Decidir la convención antes que nada.** BL-42 ya la declaró y va en dirección **contraria** a
   la conversión ×100 ingenua: *"en DB todo retorno DECIMAL (0.01 = 1 pct); nombres
   `return_decimal`/`drawdown_decimal`; el formateo a pct sólo en frontend."* Bajo esa regla el
   trabajo **no es multiplicar los datos por 100**, sino **renombrar las 10 columnas a `_decimal`**
   (dato intacto, riesgo de doble conversión nulo) y **arreglar las 5 que hoy están en pp**
   (`predicted_return_pct`, `cumulative_pnl_pct`, `running_max_dd_pct`, `running_da_*`) —
   que es el conjunto contrario al que sugiere la lectura ingenua del inventario.
   **Esta es la decisión bifurcante y debe tomarse explícitamente, no por defecto.**
2. **Unificar los escritores en desacuerdo (§3.4) pase lo que pase.** Con cualquiera de las dos
   convenciones, `cumulative_pnl_pct` y `running_max_dd_pct` no pueden seguir teniendo dos
   productores con aritmética distinta. El corta-circuitos de drawdown de L6 debe quedar operativo.
3. **Arreglar `ON CONFLICT (signal_date)` (§3.5)** o el arreglo del productor será inobservable.
4. **Corregir los `f"...{lev_pnl:+.4f}%"` de L7** (líneas 461, 499, 599, 666): el log del operador
   miente 100× hoy, con independencia de la convención que se elija.

### 5.2 Riesgo de DOBLE CONVERSIÓN — el punto crítico

Si se elige convertir los datos ×100 y arreglar el productor, hay **tres** compensaciones ya
existentes que se convertirían en errores:

| Compensación existente | Fichero:línea | Si se migra sin tocarla |
|---|---|---|
| `hsPct * 100`, `tpPct * 100` (×4 sitios) | `app/api/production/live/route.ts:74-75, 112-113, 254-255` | `/production` mostraría **HS = 300 %**, TP = 150 % |
| `rawPnl * 100` | `app/api/production/live/route.ts:211` | PnL, equity curve, Sharpe, PF, maxDD y win rate **100× inflados** |
| `/ 100.0` en el reloj de PnL | `control_system_health.py:302-303` | pasaría a ser correcto (hoy es el que está desalineado) |

**El orden importa**: productor y lector deben cambiar en el **mismo commit**. Un despliegue
intermedio en cualquier orden produce una pantalla 100× equivocada — y `/production` es
precisamente la superficie donde se leen los KPIs del ciclo semanal.

### 5.3 Cómo distinguir filas ya convertidas de filas sin convertir

El peligro real: se arregla el productor, luego se normaliza el histórico, y las filas nuevas
—ya correctas— se multiplican por segunda vez. Para estas tablas hay tres discriminadores, de más
a menos fiable:

1. **Recalcular desde los precios (recomendado, sin ambigüedad).** `forecast_h5_executions` y
   `forecast_h5_subtrades` guardan `entry_price`, `exit_price`, `direction` y `leverage` en la
   misma fila. La conversión **no necesita leer el valor viejo en absoluto**: se reescribe
   `week_pnl_pct := 100·direction·(exit_price/entry_price−1)·leverage`. Una operación así es
   **idempotente** — correrla dos veces da el mismo resultado. Elimina la clase entera de bugs de
   doble conversión para 4 de las 10 columnas (#1, #3 parcial, #8, #9).
   *Salvedad*: la fila `id=4` (2026-01-05) muestra `week_pnl_pct = 0.006104` frente a un
   `0.006183` derivado de precios (0.08 pp de diferencia); las otras 7 casan a la sexta cifra. Esa
   fila necesita revisión manual antes de recalcularla — no asumir que el derivado es el bueno.

2. **`config_version` como marcador de procedencia.** Hoy separa limpiamente las dos familias:
   `'smart_simple_v1'` = Familia A (8 filas), `'smart_executor_h5_v1'` = Familia B (2 filas de
   `signals`). Una migración puede sellar las filas convertidas con un valor nuevo
   (p. ej. `'smart_simple_v1+pctfix'`) y filtrar por él para ser re-ejecutable con seguridad.
   *Advertencia*: `signals` id 1 y 2 ya llevan `smart_executor_h5_v1` sin haber sido convertidas —
   el marcador sirve **sólo** si se sella en la propia migración, no si se infiere del valor actual.

3. **Umbral de magnitud (último recurso, sólo para stops).** Para `hard_stop_pct`/`take_profit_pct`
   el SSOT acota el rango: HS ∈ [1 %, 3 %], TP = HS/2. Sin convertir, `max = 0.03`; convertido,
   `min = 1.0`. **No hay solape**, así que `WHERE hard_stop_pct < 1.0` identifica sin ambigüedad lo
   pendiente. **Este truco NO es válido para las columnas de PnL**: un retorno semanal genuino de
   +0.8 pp y un decimal de 0.008 conviviendo en la misma columna sí solapan, y no hay umbral que
   los separe. Por eso las columnas de PnL deben ir por la vía (1) o la (2), nunca por magnitud.

4. **No olvidar los parquet.** `data/backups/features/forecast_h5_*.parquet` están versionados en
   git y contienen los valores actuales. Una migración de DB que no los regenere deja un restore
   capaz de reintroducir los decimales silenciosamente.

### 5.4 El candado que ya existe

`tests/regression/test_return_units.py::test_forecast_h5_pct_columns_hold_percentage_points` está
marcado `@pytest.mark.xfail(strict=True)` y detecta exactamente 8 de las 10 columnas confirmadas
(excluye `effective_hs_pct`/`effective_tp_pct` por `n=2 < MIN_ROWS_FOR_MEDIAN=3`). Por ser
`strict`, **cuando la fase 2 aterrice el XPASS romperá el build** y obligará a borrar el marcador
en el mismo commit. El candado está bien puesto; este inventario sólo aporta el detalle que le
falta: qué columnas, qué escritores y qué lectores compensan.

Dos huecos del candado que conviene cerrar en el mismo trabajo:
- No cubre `effective_hs_pct`/`effective_tp_pct` (N insuficiente) — quedarían fuera del gate.
- No cubre el **desacuerdo entre escritores** (§3.4): una columna puede pasar el test de mediana y
  seguir teniendo dos productores con unidades distintas.

---

## Anexo — las 17 columnas `_pct` de `forecast_h5_*`, veredicto compacto

| Tabla | Columna | Veredicto |
|---|---|---|
| `forecast_h5_executions` | `week_pnl_pct` | ❌ decimal |
| `forecast_h5_executions` | `week_pnl_unleveraged_pct` | ⚠️ 100 % NULL — no demostrable |
| `forecast_h5_executions` | `hard_stop_pct` | ❌ decimal |
| `forecast_h5_executions` | `take_profit_pct` | ❌ decimal |
| `forecast_h5_paper_trading` | `week_pnl_pct` | ❌ decimal |
| `forecast_h5_paper_trading` | `cumulative_pnl_pct` | ✅ pp (pero 2 escritores en desacuerdo) |
| `forecast_h5_paper_trading` | `running_da_pct` | ✅ pp |
| `forecast_h5_paper_trading` | `running_da_short_pct` | ✅ pp |
| `forecast_h5_paper_trading` | `running_da_long_pct` | ⚠️ 100 % NULL — no demostrable |
| `forecast_h5_paper_trading` | `running_max_dd_pct` | ✅ pp (pero 2 escritores en desacuerdo) |
| `forecast_h5_paper_trading` | `long_pct_8w` | ⚠️ 100 % NULL (no es columna de retorno) |
| `forecast_h5_predictions` | `predicted_return_pct` | ✅ pp — **columna de control** |
| `forecast_h5_signals` | `hard_stop_pct` | ❌ decimal |
| `forecast_h5_signals` | `take_profit_pct` | ❌ decimal |
| `forecast_h5_signals` | `effective_hs_pct` | ❌ decimal (n=2) |
| `forecast_h5_signals` | `effective_tp_pct` | ❌ decimal (n=2) |
| `forecast_h5_subtrades` | `pnl_pct` | ❌ decimal |
| `forecast_h5_subtrades` | `pnl_unleveraged_pct` | ❌ decimal |
