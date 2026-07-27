---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/specs/planes/04-CTR-QLAB-FABRIC-004.md
---

# Backlog gobernado — planes ACTION/DIAGNOSTIC (CTR-QLAB-FABRIC-004)

> 35 tareas pendientes derivadas de los planes 00-04 + requisito SHAP del operador,
> cruzadas con el as-built verificado el 2026-07-27 (frontend incluido). Un MD por
> tarea; cada uno con estado actual, anclas reales, impacto frontend y verificación.
> **Todas 0 trials salvo indicación en el propio BL.**

## Olas y racional

- **Ola 1 — Honestidad frontend + CI barato**: los 4 gaps encontrados en la exploración
  (caveat sin test, Gold sin banner, paper-ledger invisible, colores en predicciones) +
  timing_ratio (Etapa 0.5) + el incidente .env que BLOQUEA push.
- **Ola 2 — Gobernanza estadística (FABRIC Etapa 1)**: "cada activo sumado con el N
  fragmentado es deuda estadística sin refinanciación" — por eso va ANTES que contratos.
- **Ola 3 — Contratos, identidad y SHAP admin**: la muralla se vuelve física (tipos,
  DB, CI) + la vista de interpretabilidad.
- **Ola 4 — Hechos y linaje**: identidad (E2) precede hechos (E3) — lección auditada.
- **Ola 5 — Libro y ejecución**: snapshot→allocator→executor; strangler COP con L7 al
  final; gate = 15 criterios de §30.
- **T — Transversales**.

## Índice

| ID | Título | Ola | Esf. | Deps |
|---|---|---|---|---|
| [BL-01](BL-01-test-caveat-forecasting.md) | Test de regresión del caveat `da-caveat` | 1 | S | — |
| [BL-02](BL-02-banner-gold-weekly-inference.md) | Banner fuerte en Gold weekly-inference | 1 | S | BL-01 (el test debe cubrir ambos modos). |
| [BL-03](BL-03-wording-probabilistico-colores.md) | Wording probabilístico y neutralización de verde/rojo en predicciones | 1 | M | BL-01/BL-02 (tests actualizados con el nuevo wording). |
| [BL-04](BL-04-unificar-caveat-legacy.md) | Unificar caveat duplicado en legacy ForecastingDashboard | 1 | S | BL-01. |
| [BL-05](BL-05-production-paper-ledger-ab.md) | ProductionView consume el paper ledger (A/B v11/v12/v14) | 1 | M | — |
| [BL-06](BL-06-ci-muralla-frontend.md) | CI muralla frontend: forecasting sin aprobar/ejecutar | 1 | S | — |
| [BL-07](BL-07-timing-ratio-etapa05.md) | Etapa 0.5: timing_ratio one-off de las 4 campeonas | 1 | S | — |
| [BL-08](BL-08-incidente-env-historial.md) | Incidente .env en historial público: rotar+purgar+privatizar | 1 | M | — (bloquea el push de TODO lo demás). |
| [BL-09](BL-09-ledger-doble-ft-at.md) | Ledger doble FT-/AT- global (registries/ledger.jsonl) | 2 | L | — |
| [BL-10](BL-10-backfill-legacy-estimate-ft.md) | Backfill legacy_estimate de FT históricos (zoos) | 2 | M | BL-09. |
| [BL-11](BL-11-familias-transversales.md) | Familias transversales de hipótesis (registries/families/) | 2 | M | BL-09. |
| [BL-12](BL-12-provenance-ft-at-adr.md) | Provenance FT→AT + enmienda constitución §2 (ADR) | 2 | S | BL-09. |
| [BL-13](BL-13-campo-surface-manifiestos.md) | Campo surface en manifiestos/registry + normalize lo respeta | 2 | S | — |
| [BL-14](BL-14-components-passport-receta.md) | Bloque components: receta congelada del predictor de v11 | 2 | M | BL-12 (herencia FT). |
| [BL-15](BL-15-contrato-forecast-output.md) | Contrato forecast_output (Py+TS) + validación en el zoo | 3 | M | BL-13. |
| [BL-16](BL-16-ci-constitucional-etapa0.md) | CI constitucional Etapa 0 (legalidad + serialización canónica) | 3 | M | BL-13. |
| [BL-17](BL-17-fingerprints-canonical-writer.md) | Identidad: fingerprints + canonical writer + spine | 3 | L | BL-16. |
| [BL-18](BL-18-catalogo-motor-metricas.md) | Catálogo de métricas + motor único + metric_event | 3 | L | BL-16. |
| [BL-19](BL-19-schema-forecast-roles-db.md) | Esquema DB forecast.* + rol forecast_writer | 3 | M | BL-15. |
| [BL-20](BL-20-admin-shap-interpretabilidad.md) | Vista admin SHAP/interpretabilidad por modelo×versión (ambas superficies) | 3 | L | BL-07 (atribución reglas); opcional BL-14 (versionado del componente). |
| [BL-21](BL-21-event-sourcing-exec.md) | Event sourcing exec.* (4 entornos) + idempotencia | 4 | L | BL-17. |
| [BL-22](BL-22-fact-position-pnl.md) | fact_position / fact_pnl + identidad contable + timing_ratio persistido | 4 | L | BL-17, BL-18, BL-21. |
| [BL-23](BL-23-backfill-anti-supervivencia.md) | Backfill anti-supervivencia (campeonas+candidatas+retiradas+baselines) | 4 | M | BL-22. |
| [BL-24](BL-24-linaje-camino-dorado.md) | Linaje nodes/edges + camino dorado + revisiones tipificadas | 4 | L | BL-17. |
| [BL-25](BL-25-monitoreo-tres-relojes.md) | Monitoreo en tres relojes (control__system_health) | 4 | M | BL-18, BL-22. |
| [BL-26](BL-26-portfolio-snapshot.md) | portfolio_snapshot: barrera temporal del libro | 5 | M | BL-15 (tipos), BL-17. |
| [BL-27](BL-27-allocator-v1-novedad.md) | Allocator v1 (inverse-vol+caps) + multiplicadores + gate de novedad | 5 | L | BL-26, BL-22. |
| [BL-28](BL-28-factories-diff-semantico.md) | Factories nuevas (data/strategy/forecast) + diff semántico | 5 | L | BL-17. |
| [BL-29](BL-29-qlab-cli-cutoff-lectura.md) | CLI qlab + cutoff impuesto por la capa de lectura | 5 | L | BL-09, BL-11, BL-19 (o vista equivalente con available_at). |
| [BL-30](BL-30-execution-service-externo.md) | Servicio de ejecución fuera de Airflow + pre-trade + kill switch independiente | 5 | L | BL-21, BL-26; gate final = 15 criterios §30. |
| [BL-31](BL-31-strangler-cop.md) | Migración strangler de USD/COP (L7 al final) | 5 | L | BL-28, BL-30, BL-17. |
| [BL-32](BL-32-passport-control-tower.md) | Passport (vista live + MV) + Control Tower | 5 | L | BL-18, BL-22, BL-24; BL-05 es el primer ladrillo. |
| [BL-33](BL-33-readiness-matrix.md) | Institutional Readiness Matrix con evidencias | T | M | —(vive de evidencias de todos los demás). |
| [BL-34](BL-34-ruta-replay.md) | Ruta /replay (alias de la sección de /dashboard) | T | S | Al final (cosmético); tras BL-05. |
| [BL-35](BL-35-dataset-uris-arista-prohibida.md) | URIs de datasets + arista prohibida forecast→allocator en parseo | 3 | M | BL-13; base para BL-28. |

## Grafo mínimo de dependencias

```text
BL-08 (.env) ── bloquea push de TODO
BL-09/10/11 ─→ BL-12 ─→ BL-29 (qlab exige ledger+familias)
BL-13 ─→ BL-06/BL-15/BL-16/BL-35 (surface antes de candados por tipo)
BL-17 ─→ BL-21/22/24/28 (identidad antes de hechos — E2→E3)
BL-07 ─→ BL-20 (atribución de reglas) ─→ BL-22 (persistencia timing_ratio)
BL-26 ─→ BL-27 ─→ BL-30 ─→ BL-31 (snapshot→allocator→executor→strangler)
BL-05 ─→ BL-32 (paper ledger visible es el primer ladrillo del Passport)
```

## Reglas de ejecución

1. Ningún BL abre celdas de hipótesis: si al implementarlo aparece una decisión de
   modelado, se PARA y se pre-registra (constitución §1-§2).
2. Cada BL cierra con su verificación ejecutada y, si toca specs, el gate
   `test_knowledge_frontmatter` verde.
3. El operador prioriza; este índice no impone orden dentro de una ola salvo el grafo.
