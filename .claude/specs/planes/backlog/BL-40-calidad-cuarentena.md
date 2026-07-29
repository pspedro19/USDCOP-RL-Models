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
Entrega parcial: `src/data_quality/rules.py`, su configuración y la migración 073 definen decisiones fail-closed y `quality.quarantine_event`. No hay consumidor productivo que escriba la cuarentena. El rango USD/MXN global actual clasifica erróneamente 59 barras legítimas de 1990 porque no contempla la redenominación.

## Qué falta exactamente
Versionar rangos por época y resolverlos por instrumento+fecha; las 59 barras de 1990 deben pasar y un 2.712 moderno debe ir a cuarentena. Después cablear raw→quality→quarantine→correction→canonical y declarar `UNAVAILABLE` para columnas fantasma/sentimiento no medido.

## Impacto frontend
/analysis deja de mostrar sentiment neutro falso (UNAVAILABLE explícito).

## Dependencias
BL-36. Las series MXN/CLP anómalas NO son features de COP (verificado: features usan DXY/WTI/VIX/EMBI) — sin riesgo para v11.

## Verificación
Query de rangos imposibles = 0 en canónicas; cuarentena poblada con las filas removidas + evento.

## Notas constitución
Cuarentena, no parches: una corrección es un EVENTO con linaje, no una edición manual.
