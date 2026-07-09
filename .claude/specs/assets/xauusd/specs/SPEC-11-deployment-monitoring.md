# SPEC-11 — Despliegue y Monitoreo

## Propósito
Llevar el sistema de validado a operativo con mínimo riesgo: paper trading con **shadow-comparison**, luego live en MT5/VPS, con monitoreo de drift, atribución en vivo y kill switch. Solo se despliega lo que pasó el gate de SPEC-09.

## Fase A — Paper trading con shadow-comparison
Antes de un dólar real:
- Correr el sistema en **demo** con el **mismo `GoldTradingEnv` + `RiskLayer` + `CostModel`** que el backtest (garantiza misma distribución).
- **Shadow:** para cada barra, comparar la señal/tamaño **live** contra lo que el backtest habría hecho con los mismos datos. **Divergencia = train/serve skew** que hay que cazar antes del live (típicamente: TZ/DST, features calculadas distinto, latencia de datos, look-ahead accidental).
- Correr ≥1–3 meses de paper hasta divergencia ~0.

```python
def shadow_compare(live_decision, backtest_decision, tol) -> DivergenceReport: ...
```

## Fase B — Live (MT5 en VPS)
- Ejecución vía `MetaTrader5` (Python API) en **VPS Windows**; broker que liste XAUUSD con spreads razonables.
- Módulo `deploy/mt5_executor.py`: traduce `(direction, size)` a órdenes MT5, gestiona posición, reconcilia fills reales vs esperados.
- **Reconciliación de datos:** verificar que los datos del broker en vivo cuadran con el pipeline (mismos precios/timestamps); discrepancias grandes → pausar.
- Sizing/breakers/blackouts los sigue calculando la MISMA capa de riesgo (SPEC-06).

## Monitoreo (dashboards + alertas)
| Monitor | Qué mide | Acción |
|---|---|---|
| **Feature drift** | PSI / KS test de features live vs distribución de train | alerta; si severo, pausar |
| **Performance vs esperado** | Sharpe/DD live vs IC del backtest (SPEC-09) | alerta si sale del IC |
| **Distribución de régimen** | frecuencia de regímenes live vs histórica | detecta cambio estructural |
| **Atribución live por régimen** | PnL por régimen en vivo | valida que el edge persiste |
| **Latencia / fills** | slippage real vs modelado | recalibrar CostModel |
| **Swap real** | costo overnight real vs modelado | recalibrar |

## Kill switch
- **Manual:** botón/CLI para aplanar todo y desactivar.
- **Automático:** cualquier breaker duro de SPEC-06 (max drawdown, vol-spike sostenido) desactiva y exige revisión manual.
- **Sanity check en cada cambio de versión:** al actualizar modelo/código, reejecutar el walk-forward final y confirmar que reproduce lo anterior antes de promover (evita regresiones silenciosas).

## Gestión de modelos
- MLflow Model Registry: `candidate → staging → production`, con rollback.
- Reentrenamiento periódico (walk-forward, ~cada 3 meses) promueve nuevo candidato solo si pasa el gate (SPEC-09) y el shadow.

## Visibilidad y promoción en el frontend (ver SPEC-12)
El operador ve y promueve versiones **desde el dashboard**, sin cambios de código de frontend:
- **Entrenar una versión nueva** (hiperparámetros/features → `model_version` nueva) publica un **bundle inmutable** (`backtests/<version>/<year>`) y una entrada en `registry.json`. El front lee `/api/registry` y arma solo el selector **Activo→Estrategia→Versión→Año** con **replay** por `(strategy_id, model_version, year)`.
- **Promoción humana (Vote 2):** el operador revisa KPIs/gates/replay en `/dashboard` y aprueba/rechaza (reutiliza el flujo `approval_state.json` existente). Solo un bundle **APPROVED** pasa a producción/live.
- **Coexistencia:** v1 y v2 del mismo año son replayables lado a lado — comparar antes de promover (R8).
- **Aditivo (no romper consumos):** el registro y sus rutas (`/api/registry`, `registry.json`, `manifest.json`) se agregan **al lado** de los consumos vigentes; los endpoints/JSON legacy siguen como fallback durante la migración. Cambio de contrato = ambos mirrors TS+Python + ventana de compatibilidad. Reconciliar el drift TS↔Python del feature-contract (`architecture-overview.md` §5.3) antes de multiplicar estrategias.
- El **sanity check de versión** de abajo se ata a la inmutabilidad: reejecutar el walk-forward de la versión promovida debe reproducir su bundle publicado.

## Criterios de aceptación
- [ ] Shadow-compare implementado; corre en paper y reporta divergencia señal-a-señal.
- [ ] Divergencia paper→backtest ~0 antes de habilitar live (gate de Fase A).
- [ ] `mt5_executor` traduce decisiones a órdenes y reconcilia fills (test en demo).
- [ ] Monitores de drift (PSI/KS), performance-vs-IC y atribución live activos con alertas.
- [ ] Kill switch manual y automático (test: breaker dispara desactivación).
- [ ] Sanity check de versión obligatorio en el pipeline de promoción.
- [ ] Registry con estados y rollback probado.
- [ ] **Front dinámico:** entrenar una `model_version` nueva aparece como opción en el selector y es replayable, sin tocar código de frontend (SPEC-12: R6/R7/R8).
- [ ] **Aditivo:** los consumos front↔back existentes (`lib/contracts/*.ts`, rutas `app/api/**`, `public/data/production/*.json`) quedan **intactos**; solo se agregan rutas/archivos.

## Dependencias
SPEC-05/06 (motor de decisión), SPEC-08 (registry MLflow), SPEC-09 (gate + `register_bundle`), SPEC-10 (DAG 5 + fábrica). **SPEC-12** (registro dinámico, versionado inmutable, replay, contratos aditivos).
