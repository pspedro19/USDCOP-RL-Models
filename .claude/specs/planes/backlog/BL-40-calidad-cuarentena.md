---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - src/data_quality/ohlcv_validators.py
  - airflow/dags/l0_macro_update.py
---

# BL-40 — Calidad: cuarentena de anomalías + columnas fantasma

**Fuente**: Plan Consolidado §5 / DATA-STRATEGY §50 · **Ola**: 1-2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
Entrega parcial: `src/data_quality/rules.py`, su configuración y la migración 073 definen decisiones fail-closed y `quality.quarantine_event`. No hay consumidor productivo que escriba la cuarentena. Sobre el parquet real, el rango USD/MXN `[5,100]` rechaza 59 barras legítimas entre 1990-02 y 1995-01 (mínimo 2.712); la serie está retroajustada y no presenta salto de redenominación. Además 233.784 de 286.428 barras corresponden a instrumentos sin rango declarado y caerían como `bar.unknown_instrument`.

## Qué falta exactamente
Ampliar de forma factual USD/MXN a `[2.5,100]` (cero de las 52.644 filas USD/MXN quedarían fuera), declarar rangos para los instrumentos canónicos reales y resolver aliases antes de evaluar. Después cablear raw→quality→quarantine→correction→canonical y declarar `UNAVAILABLE` para columnas fantasma/sentimiento no medido.

## Impacto frontend
/analysis deja de mostrar sentiment neutro falso (UNAVAILABLE explícito).

## Dependencias
BL-36. Las series MXN/CLP anómalas NO son features de COP (verificado: features usan DXY/WTI/VIX/EMBI) — sin riesgo para v11.

## Verificación
Query de rangos imposibles = 0 en canónicas; cuarentena poblada con las filas removidas + evento.

## Notas constitución
Cuarentena, no parches: una corrección es un EVENTO con linaje, no una edición manual.
