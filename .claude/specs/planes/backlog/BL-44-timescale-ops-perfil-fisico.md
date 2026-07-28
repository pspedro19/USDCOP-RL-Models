---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - docker-compose.yml
  - database/migrations/067_spx500_regime_macro_vars.sql
---

# BL-44 — TimescaleDB ops + perfil físico ampliado

**Fuente**: Plan Consolidado §10 / DATA-STRATEGY §51 (D8/P2) · **Ola**: 5 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
7 hypertables; el perfil actual NO expone chunk interval, compresión, retención, índices, caggs, tamaño físico ni bloat. Hypertables VACÍAS creadas prematuramente (crypto_exposure_signals, flows, onchain).

## Qué falta exactamente
Política: chunk por event_time, segmentby instrument_id; caggs 1h/4h/1d (BL-38); histórico frío a MinIO/Parquet ANTES de cualquier retención; el próximo perfil añade tamaño físico, PK/FK, políticas, gaps, clasificación SSOT/projection/cache/deprecated; NO crear hypertables para tablas pequeñas o vacías.

## Impacto frontend
Ninguno.

## Dependencias
BL-38; después de BL-36 (no optimizar lo que se va a retirar).

## Verificación
Perfil v2 con las columnas nuevas; compresión medida en las 3 tablas grandes.

## Notas constitución
Infraestructura proporcional a la escala: optimizar 2.2M filas es mantenimiento, no urgencia.
