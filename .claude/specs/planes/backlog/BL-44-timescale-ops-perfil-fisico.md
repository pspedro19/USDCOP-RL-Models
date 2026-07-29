---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - docker-compose.yml
  - database/migrations/067_spx500_regime_macro_vars.sql
---

# BL-44 — TimescaleDB ops + perfil físico ampliado

**Fuente**: Plan Consolidado §10 / DATA-STRATEGY §51 (D8/P2) · **Ola**: 5 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
Entrega parcial: la migración 080 registra intención física, exige escala/evidencia de restore y ofrece funciones operator-only para convertir tablas e instalar caggs. Nada de ello está aplicado: la base viva conserva 7/7 hypertables sin compresión, cero caggs y tres hypertables vacías creadas prematuramente.

## Qué falta exactamente
Ejecutar el preflight/operator function sobre las tablas grandes, instalar y medir compresión/caggs con catálogos TimescaleDB 2.x, y retirar las hypertables vacías sólo mediante una migración revisada. Falta un perfil v2 ejecutable con tamaño antes/después; ninguna prueba puede sustituir que la compresión exista realmente.

## Impacto frontend
Ninguno.

## Dependencias
BL-38; después de BL-36 (no optimizar lo que se va a retirar).

## Verificación
Perfil v2 con las columnas nuevas; compresión medida en las 3 tablas grandes.

## Notas constitución
Infraestructura proporcional a la escala: optimizar 2.2M filas es mantenimiento, no urgencia.
