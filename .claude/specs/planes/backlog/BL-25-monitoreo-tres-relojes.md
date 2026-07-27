---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
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

## Notas constitución
El retiro se dispara por protocolo, nunca por cómo se sienta el mes.
