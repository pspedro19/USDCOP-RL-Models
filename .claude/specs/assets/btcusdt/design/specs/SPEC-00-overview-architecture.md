# SPEC-00 — Overview & Arquitectura de la Capa de Modelado

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 (basada en Guía Maestra v3) |
| Depende de | Pipeline de datos `btc_pipeline` (specs 00–06): `btc_daily.parquet` + `btc_price_1h.parquet` |
| Materializa | `strategy/__init__.py`, contrato del orquestador, diagrama de módulos |
| Pre-registro | Toda la Guía v3 |

## 1. Propósito y alcance

Definir la **arquitectura global** de la capa de modelado: las cinco capas, el flujo de
datos entre ellas, el contrato del orquestador, y el mapa de módulos. Es el índice
arquitectónico; cada componente tiene su propia spec (SPEC-01…12).

**Fuera de alcance:** la ingeniería del dataset (pipeline `00–06`), y la operación del
capital real más allá de lo que especifica SPEC-12.

## 2. Entradas (contrato de entrada del sistema)

El sistema consume **exclusivamente** el output del pipeline de datos. No accede a APIs
externas en runtime de decisión (eso es responsabilidad del ETL, que corre antes).

| Insumo | Tipo | Fuente | Disponibilidad | Contrato |
|---|---|---|---|---|
| `btc_daily` | `pd.DataFrame` | `btc_pipeline` | barra UTC 00:00, cerrada | índice `DatetimeIndex` UTC tz-naive, único, ordenado; ~44 columnas del esquema SSOT |
| `btc_price_1h` | `pd.DataFrame` | `btc_pipeline` | barra horaria cerrada | OHLCV + CVD/OFI; para ejecución y breaker |
| `funding` | columna de `btc_daily` | Binance perp | D−1 | float, señal (no costo) |
| `etf_flows` | columna de `btc_daily` | Farside | **D+1** | float, lag estricto |
| `headlines_stream` | cola de eventos | archivo de titulares (§7.8) | tiempo real | texto tiempo-de-titular + timestamp |

**Precondición dura:** el dataset debe haber pasado el QA bloqueante del pipeline (I1–I10).
Si `QUALITY_FLAG` está activo en una barra, esa barra no participa del entrenamiento.

## 3. Salidas (contrato de salida del sistema)

| Salida | Tipo | Rango | Descripción |
|---|---|---|---|
| `exposure_t` | `float` | **[0.0, 1.0]** | Fracción de NAV en BTC spot, tras todos los gates y el motor |
| `target_vs_current` | `float` | [−1, 1] | Delta que dispararía (o no) el rebalanceo |
| `rebalance_flag` | `bool` | — | `True` solo si se cruzó la banda ±12.5 % y pasó el meta-modelo |
| `decision_trace` | `dict` | — | Estado de cada gate (`z_ciclo`, `z_funding`, `M_interno`, `M_liquidez`, `G_eventos`, régimen HMM) para auditoría |

**Invariante de salida global:** `0.0 ≤ exposure_t ≤ 1.0` en TODA barra, sin excepción
(spot-only). Cualquier valor fuera de rango es un fallo crítico que detiene el sistema.

## 4. Interfaz (API)

```python
from typing import Protocol
import pandas as pd

class StrategyEngine(Protocol):
    def decide(self, daily: pd.DataFrame, hourly: pd.DataFrame,
               events: list[Event], state: EngineState) -> Decision:
        """Devuelve la decisión de exposición para la última barra cerrada.
        NO ejecuta órdenes; solo computa exposure_t + trace. Determinista dado el input."""
        ...

@dataclass(frozen=True)
class Decision:
    exposure: float               # [0, 1]
    rebalance: bool
    target_vs_current: float
    trace: dict                   # decision_trace
```

Contrato: `decide` es **puro y determinista** dado `(daily, hourly, events, state)`. Ninguna
llamada de red, ninguna lectura de reloj salvo la barra provista. Testeable offline.

## 5. Arquitectura de cinco capas (flujo de datos)

```
btc_daily ─┬─► [SPEC-01] Régimen HMM (fit congelado) ──► z_ciclo ──┐
           │                                                        ├─► [SPEC-03] R = 0.7·z_ciclo + 0.3·z_funding
           ├─► [SPEC-02] Posicionamiento ───────────► z_funding ───┘         └─► M_interno ∈ [0.25, 1.0]
           │
           ├─► [SPEC-04] Liquidez ─────────────────────────────────────────► M_liquidez ∈ [0.5, 1.25]
           │
headlines ─┴─► [SPEC-05] Eventos (LLM + vol-spike breaker) ─────────────────► G_eventos ∈ {1, .5, .25, 0}
                                                                                        │
btc_daily ─────► [SPEC-06] Motor: vol_target(σ_down,30%) × M_interno × M_liquidez × G_eventos
                            + bandas ±12.5% + mínimo 24h + límites duros + cap portafolio
                                                                                        │
                                                                                        ▼
                            [SPEC-07] Ejecución & CostModel ──► orden (si rebalance_flag)
                                                                                        │
btc_price_1h ─► [SPEC-08] Meta-labeling (frena trades) ─► [SPEC-09] RL táctico (±10% NAV, opcional)
                                                                                        │
                            [SPEC-11] Validación envuelve TODO (walk-forward, DSR, atribución)
                            [SPEC-12] Retiro monitorea el sistema en vivo
```

**Regla de combinación (SPEC-03):** ciclo y funding se **combinan en riesgo** (correlacionados);
liquidez y eventos **multiplican** (semi-ortogonal / ortogonal). La ortogonalidad se mide
trimestralmente (H-CMB-02).

## 6. Mapa de módulos

```
strategy/
├── __init__.py            # orquestador StrategyEngine (esta spec)
├── regime.py              # SPEC-01
├── positioning.py         # SPEC-02
├── risk_combination.py    # SPEC-03  *
├── liquidity.py           # SPEC-04
├── events.py              # SPEC-05  (classifier + vol_spike_breaker)
├── engine.py              # SPEC-06  (vol-targeting, bandas, motor)
├── execution.py           # SPEC-07  (CostModel, órdenes)
├── meta_label.py          # SPEC-08
├── baselines.py           # SPEC-10  (B1, B2, S3, S4, S5)
└── rl/                    # SPEC-09  (entorno Gymnasium + PPO)
validation/                # SPEC-11  (walk-forward, purge/embargo, DSR, bootstrap, atribución)
ops/
└── retirement.py          # SPEC-12  (Retirement Monitor)
```

## 7. Orden de construcción (roadmap — el gate de cada fase es una hipótesis del registro)

| Fase | Entregable | Gate (hipótesis) |
|---|---|---|
| 3 | SPEC-01 régimen fit congelado | H-REG-01, H-REG-02 |
| 4 | SPEC-02 posicionamiento | H-POS-01 |
| 5 | SPEC-03 combinación en riesgo | **H-CMB-01, H-CMB-02** |
| 6 | SPEC-04 liquidez | H-LIQ-01, H-LIQ-02 |
| 7 | SPEC-06 motor = **S3** | **H-ENG-01, H-ENG-02** |
| 8 | SPEC-05 eventos | H-EVT-01, H-EVT-02 |
| 9 | SPEC-08 meta-labeling = **S4** | H-MET-01, H-MET-02 |
| 10 | SPEC-09 RL = **S5** (opcional) | H-RL-01 |
| 11 | SPEC-12 retiro + shadow | protocolo §14 firmado |

> **B1 y B2 (SPEC-10) se construyen en Fase 2, ANTES de todo lo demás.** Sin benchmark no hay
> gate.

## 8. Invariantes del sistema

1. `exposure ∈ [0, 1]` siempre (spot-only).
2. Ninguna decisión usa datos posteriores a la barra cerrada (anti-look-ahead capa datos).
3. Los labels de régimen son replay como-en-vivo (anti-look-ahead capa modelos, SPEC-01).
4. El recall del clasificador se reporta como cota superior (anti-look-ahead capa
   clasificadores, SPEC-05).
5. Ningún componente entra sin rechazar su H0 (default = descartar).
6. El cap de portafolio es inmune a los gates.

## 9. Test de hipótesis (nivel sistema)

**H0 (H-SYS-01):** el sistema final tiene Sharpe ≤ 0 tras deflación.
**Estadístico:** Deflated Sharpe Ratio con `N_trials_total` del registro de hipótesis.
**Criterio:** DSR > 0 con confianza ≥ 95 %. Ver SPEC-11 §9.

## 10. Protocolo de backtest (referencia)

Todo backtest de cualquier capa usa el motor de SPEC-11: walk-forward con purga/embargo,
CostModel de SPEC-07, replay de labels de SPEC-01, reporte por sub-era, sensibilidades
completas. Ningún componente se evalúa con un split ingenuo.

## 11. Criterios de aceptación (DoD del sistema)

- [ ] Cada SPEC-01…12 tiene su DoD cumplido (constitución §10).
- [ ] La cadena completa produce `exposure ∈ [0,1]` sobre todo el histórico sin violar
      invariantes.
- [ ] S3 rechaza H-ENG-01 y H-ENG-02 (le gana a B1 y B2 en Calmar OOS).
- [ ] El DSR del sistema (H-SYS-01) es positivo con el conteo completo de trials.
- [ ] El protocolo de retiro está firmado.
- [ ] Aceptado por escrito que el resultado válido puede ser "S3 ES la estrategia".

## 12. Anti-look-ahead / modos de fallo cubiertos

Esta spec es el índice; cada modo de fallo del stress-test (§7.1–7.8) se resuelve en su spec:
7.1 doble conteo → SPEC-03 · 7.2 apalancamiento → SPEC-06 (spot-only) · 7.3 turnover →
SPEC-07 · 7.4 selección → HYPOTHESIS-REGISTRY + SPEC-11 · 7.5 asimetría eventos → SPEC-05 ·
7.6 supervivencia → SPEC-06 (cap portafolio) · 7.7 historia corta → SPEC-01 · 7.8
contaminación LLM → SPEC-05 + ADR-0011.

## 13. Parámetros pre-registrados (aplicables)

Todos los de `PRE-REGISTRATION.md`. Esta spec no introduce parámetros nuevos; orquesta los
de las specs hijas.
