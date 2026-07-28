---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/public/data/registry.json
  - scripts/pipeline/normalize_champions.py
---

# BL-43 — Aislar el modelo sintético demo (CI que lo bloquee fuera de demo)

**Fuente**: Plan Consolidado §7 (final) · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
config.models contiene investor_demo con algorithm=SYNTHETIC (NOT for live trading) y status=active — una demo sintética puede colarse en métricas/superficies reales.

## Qué falta exactamente
environment=demo + surface=synthetic + execution_eligible=false; mover a demo.*; CI que RECHACE algorithm=SYNTHETIC con environment distinto de demo; verificación de que ninguna vista de performance real lo lista.

## Impacto frontend
Si alguna vista lo muestra, gana badge DEMO inequívoco o desaparece.

## Dependencias
BL-13 (campo surface).

## Verificación
Test rojo con SYNTHETIC+active fuera de demo.

## Notas constitución
Desconfianza de la magia: un equity sintético presentado como real es el peor bug de honestidad posible.
