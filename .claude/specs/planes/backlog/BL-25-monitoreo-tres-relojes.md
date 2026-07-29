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
