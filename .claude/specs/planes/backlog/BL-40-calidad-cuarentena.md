---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - src/data_quality/ohlcv_validators.py
  - airflow/dags/l0_macro_update.py
---

# BL-40 — Calidad: cuarentena de anomalías + columnas fantasma

**Fuente**: Plan Consolidado §5 / DATA-STRATEGY §50 · **Ola**: 1-2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
Anomalías REALES del perfil: USD/MXN max 175,814 (mediana 19.65), USD/CLP max 94,890 (mediana 870) — parsing de miles o columna equivocada. Columnas fantasma: forwards.forward_rate 100 pct NULL (por diseño: declarar), crypto liquidations_usd 100 pct NULL, OI 44/2506, news sentiment_score=0 y label=neutral PARA TODOS (el motor no corre) + content/gdelt_tone/entities 100 pct NULL.

## Qué falta exactamente
Pipeline raw → quality FAIL → quarantine → comparación proveedor → correction event → canónica (jamás UPDATE a mano); reglas de rango versionadas (mxn [5,100], clp [100,5000] — amplias, no recortan movimientos reales); feature_status=UNAVAILABLE para fantasmas (no publicar ceros que parezcan medición); arreglar o apagar el motor de sentimiento de news.

## Impacto frontend
/analysis deja de mostrar sentiment neutro falso (UNAVAILABLE explícito).

## Dependencias
BL-36. Las series MXN/CLP anómalas NO son features de COP (verificado: features usan DXY/WTI/VIX/EMBI) — sin riesgo para v11.

## Verificación
Query de rangos imposibles = 0 en canónicas; cuarentena poblada con las filas removidas + evento.

## Notas constitución
Cuarentena, no parches: una corrección es un EVENTO con linaje, no una edición manual.
