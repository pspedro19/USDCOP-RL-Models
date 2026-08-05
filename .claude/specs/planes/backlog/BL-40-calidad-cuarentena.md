---
kind: roadmap
status: PARTIAL
version: 1.3.2
last_verified: 2026-08-05
supersedes: []
code_anchors:
  - src/data_quality/ohlcv_validators.py
  - src/data_quality/rules.py
  - config/quality/market_price_ranges.yaml
  - src/market/publication.py
  - src/data_quality/corrections.py
  - src/data_quality/feature_availability.py
  - airflow/dags/l0_ohlcv_realtime.py
  - airflow/dags/l0_ohlcv_backfill.py
  - airflow/dags/news_daily_pipeline.py
---

# BL-40 — Calidad: cuarentena de anomalías + columnas fantasma

**Fuente**: Plan Consolidado §5 / DATA-STRATEGY §50 · **Ola**: 1-2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-08-05)

Implementación funcional cerrada, con verificación productiva durable pendiente.
`src/data_quality/rules.py`, su configuración y las migraciones 072, 073 y 084
definen identidad, decisiones fail-closed, cuarentena y correcciones con contexto causal. El
rango USD/MXN dejó de ser universal: la regla está limitada al productor `twelvedata` y a
observaciones desde `1993-01-01T00:00:00Z`, corte de unidad monetaria documentado por Banxico
SIE CF373.

Los productores realtime y backfill llaman `publish_or_declare_gap` antes de escribir; el
ingestor genérico usa el mismo publicador cuando la identidad tiene una regla escalonada. Una
barra rechazada se persiste en `quality.quarantine_event` y no llega a
`market.canonical_bar`. `apply_market_correction` reevalúa con el instante original, enlaza un
único evento de corrección y publica a canonical antes de marcar `CORRECTED`; los reintentos
idénticos son idempotentes y una corrección diferente falla cerrada.

La medición de columnas fantasma se ejecuta tras cada corrida de noticias con el
`data_interval_end` exacto. El análisis consume el último corte de las 18:00 UTC y convierte
sentimiento no medido o placeholder constante en `null` más `reason`, nunca en neutral falso.

El evaluador exige alias canónico, proveedor y timestamp timezone-aware. Proveedor distinto,
instante anterior al corte, contexto ausente o llamada directa sin contexto quedan en
`QUARANTINED`; no se recortan ni corrigen precios silenciosamente. Los rangos legacy no scoped
siguen admitidos para no inventar proveedor/fecha de instrumentos aún no migrados.

Cuando un proveedor tiene varios regímenes, se aplica el corte `valid_from` más reciente que ya
esté vigente. Dos reglas del mismo proveedor con el mismo corte se rechazan al cargar la
configuración; no se resuelven por orden accidental del YAML.

## Alcance cerrado y límites deliberados

- Las reglas se activan únicamente donde hay proveedor, símbolo e intervalo resolubles y una
  regla escalonada. Instrumentos con rangos legacy o sin evidencia declaran el hueco y conservan
  el camino anterior; no se inventa cobertura.
- Las migraciones gobernadas están aplicadas en PostgreSQL. La migración 084 es aditiva y
  preserva los eventos legacy, que no pueden corregirse sin el contexto original.
- Los probes de corrección usan una transacción exterior y terminan en rollback. Por eso la
  tabla puede quedar vacía después de verificar el ciclo completo; no se dejan eventos de prueba
  haciéndose pasar por incidentes reales.
- Los DAGs que materializan barras canónicas y cuarentenas permanecen pausados. Mientras
  `market.canonical_bar` y `quality.quarantine_event` sigan vacías, el criterio productivo de
  rangos imposibles y cuarentena poblada se cumple por vacuidad y no habilita promoción.
- Tres ventanas controladas demostraron y corrigieron, en orden, alcance decorativo, cascada de
  `skipped` y fan-in incompleto. La tercera firmó en el scheduler el alcance correcto
  (COP/BRL `skipped`, MXN/export/validate ejecutados), pero los dos fetch MXN recibieron `401` y
  el camino antiguo los reportó como `SUCCESS`; `b96ce054` inicia el cierre fail-closed de esa
  mentira. No se abre otra ventana mientras no exista una fuente autenticada y el cross-review
  del fallo parcial no esté cerrado.

## Impacto frontend
/analysis deja de mostrar sentiment neutro falso (UNAVAILABLE explícito).

## Dependencias
BL-36 y aplicación autorizada de las migraciones 072/073 de `fabric-v1`. Las series MXN/CLP
anómalas NO son features de COP (verificado: features usan DXY/WTI/VIX/EMBI) — sin riesgo para
v11.

## Verificación

- Suite focal conjunta de productores, publicador, corrección y disponibilidad: `60 passed`.
- PostgreSQL: query de rangos imposibles USD/MXN posteriores al corte = `0` en canonical.
- Probe reversible PostgreSQL: barra inválida → cuarentena contextual; corrección válida →
  canonical + evento + estado `CORRECTED`; retry idéntico idempotente; rollback exterior deja
  el estado previo intacto.
- Tarea C028 real: cortes diarios distintos y exactos; al corte de las 18:00 UTC se observaron
  `7/7 UNAVAILABLE`. El consumidor real devolvió `null+reason` para las filas afectadas.
- Pendiente para DONE bilateral: una ventana productiva autorizada y fechada para ejecutar un
  productor real, seguida de evidencia durable de barras aceptadas en canonical y al menos una
  barra rechazada con su evento de cuarentena. Los probes con rollback verifican la mecánica,
  pero no sustituyen este criterio operativo.
- **Bloqueo externo medido:** el proveedor configurado rechazó las solicitudes con `401`; no se
  insertó ninguna barra y Fabric permaneció vacío. Resolver requiere provisión de credenciales
  reales por Vault o una fuente alternativa gobernada. No se copian claves al repositorio ni se
  repite el run contra una autenticación conocida como inválida. Hasta esa decisión BL-40 sigue
  `PARTIAL` bloqueado por dato/autenticación, no por una promoción pendiente.

## Notas constitución
Cuarentena, no parches: una corrección es un EVENTO con linaje, no una edición manual.
