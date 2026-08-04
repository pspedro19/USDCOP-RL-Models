---
kind: roadmap
status: PARTIAL
version: 1.2.0
last_verified: 2026-08-03
supersedes: []
code_anchors:
  - src/data_quality/ohlcv_validators.py
  - src/data_quality/rules.py
  - config/quality/market_price_ranges.yaml
  - airflow/dags/l0_macro_update.py
---

# BL-40 — Calidad: cuarentena de anomalías + columnas fantasma

**Fuente**: Plan Consolidado §5 / DATA-STRATEGY §50 · **Ola**: 1-2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-08-03)

Entrega parcial: `src/data_quality/rules.py`, su configuración y la migración 073 definen
decisiones fail-closed y `quality.quarantine_event`. El rango USD/MXN dejó de ser universal:
la regla está limitada al productor `twelvedata` y a observaciones desde
`1993-01-01T00:00:00Z`, corte de unidad monetaria documentado por Banxico SIE CF373. Falta
todavía un consumidor productivo que invoque la regla y escriba la cuarentena.

El evaluador exige alias canónico, proveedor y timestamp timezone-aware. Proveedor distinto,
instante anterior al corte, contexto ausente o llamada directa sin contexto quedan en
`QUARANTINED`; no se recortan ni corrigen precios silenciosamente. Los rangos legacy no scoped
siguen admitidos para no inventar proveedor/fecha de instrumentos aún no migrados.

Cuando un proveedor tiene varios regímenes, se aplica el corte `valid_from` más reciente que ya
esté vigente. Dos reglas del mismo proveedor con el mismo corte se rechazan al cargar la
configuración; no se resuelven por orden accidental del YAML.

## Qué falta exactamente

- Aplicar y poblar primero la identidad `reference.provider_symbol` (migración 072) y el
  sumidero `quality.quarantine_event` (073). Sin ambos, cablear el gate convertiría una
  cuarentena fail-closed en pérdida silenciosa de barras.
- Después, cablear `evaluate_provider_bar` en los productores realtime/backfill exclusivamente
  para USD/MXN antes del upsert y persistir cada rechazo antes de impedir que llegue a canonical.
  Esos DAGs COP son ownership de CLAUDE y requieren incremento coordinado.
- Declarar reglas scoped para los demás instrumentos sólo con proveedor, ventana y fuente
  verificables; un rango legacy no basta para cierre y el gate no debe activarse para COP/BRL
  mientras esa identidad no exista.
- Cablear raw→quality→quarantine→correction→canonical y declarar `UNAVAILABLE` para columnas
  fantasma/sentimiento no medido.

## Impacto frontend
/analysis deja de mostrar sentiment neutro falso (UNAVAILABLE explícito).

## Dependencias
BL-36 y aplicación autorizada de las migraciones 072/073 de `fabric-v1`. Las series MXN/CLP
anómalas NO son features de COP (verificado: features usan DXY/WTI/VIX/EMBI) — sin riesgo para
v11.

## Verificación
Query de rangos imposibles = 0 en canónicas; cuarentena poblada con las filas removidas + evento.

## Notas constitución
Cuarentena, no parches: una corrección es un EVENTO con linaje, no una edición manual.
