---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - airflow/dags/core_watchdog.py
  - services/common/metrics.py
---

# BL-25 — Monitoreo en tres relojes (control__system_health)

**Fuente**: FABRIC §23 · **Ola**: 4 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Watchdog auto-heal existe (staleness operativa). No hay reloj de MODELO (PSI/KS, drift de predicción) ni de PnL (TE 3σ live-vs-paper, decay) con acciones automáticas.

## Qué falta exactamente
Motor único sobre facts+metric_event: datos(min, fail-closed/QUARANTINE), modelo(diario, PSI>0.25 congela promociones), PnL(semanal, dispara REDUCED/withdrawal). Tabla §23.1 como contrato.

## Impacto frontend
Semáforos en Control Tower/production.

## Dependencias
BL-18, BL-22.

## Verificación
Inyectar drift sintético ⇒ promoción congelada; TE>3σ ⇒ evento withdrawal.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/unit/test_system_health.py -q
verde:   25 passed

muta:    src/monitoring/system_health.py:338 — TRACKING_ERROR_SIGMA * 1000
espera:  1 failed — el gemelo de 3.50-sigma deja de disparar withdrawal (GREEN != ORANGE)

muta-2:  src/monitoring/system_health.py:338 — TRACKING_ERROR_SIGMA / 1000
espera:  1 failed — el gemelo de 2.47-sigma deja de estar verde (ORANGE != GREEN)

muta-3:  ruido del escenario -> 0.0 (vuelve a `live = paper - constante`)
espera:  1 failed — "sd(d)=5.13e-19: diferencia casi constante, z-score vacio"
```

**Historial honesto**: hasta el 2026-07-28 el umbral **no estaba anclado**. Multiplicarlo por
1000 pasaba verde, porque el fixture usaba `live = paper − 0.02` (diferencia constante): `sd(d)`
era ruido de coma flotante ~1e-18 y el z-score salía ~1e16. El test demostraba que la rama
existe, no que el umbral fuera 3. Ahora el 3 queda **acotado por arriba y por abajo**.

## Notas constitución
El retiro se dispara por protocolo, nunca por cómo se sienta el mes.

## El reloj de DATOS ya mira el linaje (2026-08-06, `8968dc73`)

**El defecto que cerró.** `evaluate_data_clock` medía frescura de **fuentes** —OHLCV m5, macro
diario, seeds— y **no consultaba `lineage.node` ni `lineage.strategy_node`**, aunque BL-24 ya había
entregado `status VALID/STALE/INVALIDATED` y los roles por estrategia. Consecuencia concreta: **un
dato fresco cuyo nodo estaba `INVALIDATED` pasaba como «ok»**. El reloj daba verde sobre linaje
degradado, que es justo lo que FABRIC §23 quiere impedir.

Ahora consulta los nodos enlazados a `H5_PRODUCTION_STRATEGY_ID` y cualquiera en
`STALE`/`INVALIDATED` produce un `DataProbe` **activo**; el motor existente lo convierte en
`FAIL_CLOSED` + `BLOCK_SIGNAL` sin cambios (el motor y el contrato **no se tocaron**).

### Sólo `INPUT` y `SIGNAL`, y es deliberado

Son los roles que el sistema **materializa hoy** — medido en la DB viva: 3 `INPUT` + 1 `SIGNAL`.
`FEATURE` y `MODEL` están en el `CHECK` de la tabla pero **nadie los enlaza**: exigirlos pondría
rojo algo que nadie ha prometido, y **un rojo falso gasta la misma credibilidad que un verde
falso** — la misma razón por la que `smart_simple_v11` quedó fuera del gate cross-SSOT. El día que
se enlacen, un candado obliga a decidirlo a conciencia en vez de heredarlo.

**Anti-vacuidad**: la ausencia de un rol produce `missing` **activo**, no silencio. Sin eso, perder
los links daría verde — el probe no encontraría nada degradado *porque no encontraría nada*.

| mutación | rojos |
|---|---|
| saltarse el rol ausente (verde por vacío) | 3 |
| marcar el probe como diagnóstico | 3 |
| ignorar `INVALIDATED` | 2 |
| interpolar el `strategy_id` en el SQL | 1 |

### Una confusión que conviene dejar escrita

`smart_simple_v11` es **dos cosas**: el id de la estrategia de **producción** del track H5
(`h5_strategy_identity.py`) y un **policy spec `SPEC_ONLY`** en `config/policies/`. Los nodos de
linaje pertenecen al **pipeline vivo**, no a la migración pendiente al motor de políticas. Ambos
agentes lo confundimos al discutir el shape, y el nombre compartido lo invita.

### Lo que SIGUE abierto

1. **Reloj de MODELO** (diario, PSI > 0.25 congela promociones) — no tocado.
2. **Reloj de PnL** (semanal, dispara `REDUCED`) — no tocado.
3. **`facts`** no existe en la DB (medido); el motor único sobre `facts + metric_event` que la
   ficha pide sigue dependiendo de esa migración.
4. `lineage.revision_event` está a **0**: el productor existe y está cableado
   (`macro_revision.py` ← `upsert_service` ← `l0_macro_update`, schedule `0 13-17 * * 1-5`), pero
   **aún no se ha observado ninguna revisión real**. El probe está probado contra transiciones que
   el sistema sabe hacer, no contra transiciones que ya haya hecho.

**BL-25 sigue `PARTIAL`**: esto cierra el reloj de datos en su dimensión de linaje, no los tres
relojes.
