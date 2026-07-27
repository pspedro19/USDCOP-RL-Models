---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - services/signalbridge_api/app/services/pretrade.py
  - src/risk
---

# BL-30 — Servicio de ejecución fuera de Airflow + pre-trade + kill switch independiente

**Fuente**: FABRIC §21 + §30 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
PreTradeGate paper-first y kill-switch existen DENTRO de SignalBridge; falta: consumo exclusivo de portfolio_target, reconciliación pre-operación, kill switch verificado con Airflow caído.

## Qué falta exactamente
Servicio consume SOLO control.portfolio_target (jamás señales sueltas); checklist §21.1 por orden; reconciliación pre/intra/EOD; kill switch niveles BLOCK_NEW/CANCEL_OPEN/EXIT_ALL/FREEZE consultado antes de cada apertura; credenciales fuera de Airflow/repo/frontend.

## Impacto frontend
/execution muestra niveles del kill switch y reconciliación.

## Dependencias
BL-21, BL-26; gate final = 15 criterios §30.

## Verificación
Simulacro: Airflow apagado ⇒ kill switch operativo; discrepancia broker ⇒ QUARANTINED.

## Notas constitución
Airflow publica targets; NUNCA envía órdenes.
