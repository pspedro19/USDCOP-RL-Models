---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - tests/regression/test_strategy_manifests.py
---

# BL-16 — CI constitucional Etapa 0 (legalidad + serialización canónica)

**Fuente**: FABRIC §8.3, §11.2, §28 E0 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Piezas sueltas existen (safe_json_dump sin NaN/Inf, manifest drift test). Falta la matriz de legalidad y la política canónica como validadores.

## Qué falta exactamente
Validador de declaraciones: matriz research_state×capital_tier×operational_state; serialización canónica (UTF-8 NFC, claves ordenadas, ISO-Z, decimales cuantizados); 'el CI rechaza una declaración inválida ANTES de ejecutar un DAG'.

## Impacto frontend
Ninguno.

## Dependencias
BL-13.

## Verificación
Declaración PAPER+FULL ⇒ CI rojo; JSON con NaN ⇒ rojo.

## Notas constitución
96 combinaciones nominales, la mayoría absurdas — la matriz es la defensa.
