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

> 47 tareas pendientes derivadas de los planes 00-04 + requisito SHAP del operador,
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
| [BL-36](BL-36-racionalizacion-inventario-db.md) | Racionalización del inventario DB (59 tablas → matriz de verdad) | 2-3 | M | antes de BL-15/18/19/21/22 |
| [BL-37](BL-37-identidades-canonicas.md) | Identidades canónicas (reference.asset/instrument/provider_symbol/bar_ | 2-3 | M | ver BL |
| [BL-38](BL-38-market-canonical-resampleo.md) | Mercado canónico: raw_bar/canonical_bar + caggs 1h/4h/1d + política de | 3-4 | L | ver BL |
| [BL-39](BL-39-feature-contracts-normalizacion.md) | Feature contracts por estrategia-versión + normalización al artefacto | 2-3 | L | ver BL |
| [BL-40](BL-40-calidad-cuarentena.md) | Calidad: cuarentena de anomalías + columnas fantasma | 1-2 | M | ver BL |
| [BL-41](BL-41-seguridad-db-p0.md) | Seguridad DB P0: secret.*, credenciales consolidadas, timestamptz | 1 | M | ver BL |
| [BL-42](BL-42-unidades-decimales-signal-normalizada.md) | Unidades decimales + action.strategy_signal normalizada (JSONB de polí | 3 | M | ver BL |
| [BL-43](BL-43-demo-sintetica-aislada.md) | Aislar el modelo sintético demo (CI que lo bloquee fuera de demo) | 1 | S | ver BL |
| [BL-44](BL-44-timescale-ops-perfil-fisico.md) | TimescaleDB ops + perfil físico ampliado | 5 | M | ver BL |
| [BL-45](BL-45-policy-engine-contrato.md) | Motor de políticas: contrato + registry + factory (R1-R3) | 3 | L | ver BL |
| [BL-46](BL-46-policy-backend-frontend.md) | Políticas: backend (policy_version/signal) + frontend schema-driven (R | 3-4 | L | ver BL |
| [BL-47](BL-47-policy-migracion-r6-r8.md) | Migración de estrategias al motor de políticas (R6-R8) | 5 | L | ver BL |

## Grafo mínimo de dependencias

> Fuentes nuevas 2026-07-27: `Plan_Consolidado_usdcop_trading.md` (P0/P1/P2 de DB)
> y `CTR-QLAB-FABRIC-004-DATA-STRATEGY.md` PARTE II §33-57 (D0-D8 de datos/features)
> → BL-37..44. `05-rule-based-strategies.md` (motor de políticas R1-R8,
> invariantes promovidas a `.claude/rules/strategy-engines.md`) → BL-45..47. Ampliaciones derivadas: BL-24 gana `availability_quality` (solo 9.1%
> del PIT es vintage real; `publication_date` 100% NULL en monthly/quarterly ⇒
> BLOQUEA promotion) y BL-36 gana staging_contract + semántica de 5 timestamps.

```text
BL-08 (.env) ── bloquea push de TODO
BL-09/10/11 ─→ BL-12 ─→ BL-29 (qlab exige ledger+familias)
BL-36 ─→ BL-15/18/19/21/22 (decidir destino antes de crear esquemas)
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

## Documentos de este directorio

<!-- idx:auto -->

| Documento | Estado |
|---|---|
| [BL-01 — Test de regresión del caveat da-caveat](BL-01-test-caveat-forecasting.md) | IMPLEMENTED |
| [BL-02 — Banner fuerte en Gold weekly-inference](BL-02-banner-gold-weekly-inference.md) | IMPLEMENTED |
| [BL-03 — Wording probabilístico y neutralización de verde/rojo en predicciones](BL-03-wording-probabilistico-colores.md) | IMPLEMENTED |
| [BL-04 — Unificar caveat duplicado en legacy ForecastingDashboard](BL-04-unificar-caveat-legacy.md) | IMPLEMENTED |
| [BL-05 — ProductionView consume el paper ledger (A/B v11/v12/v14)](BL-05-production-paper-ledger-ab.md) | IMPLEMENTED |
| [BL-06 — CI muralla frontend: forecasting sin aprobar/ejecutar](BL-06-ci-muralla-frontend.md) | IMPLEMENTED |
| [BL-07 — Etapa 0.5: timing_ratio one-off de las 4 campeonas](BL-07-timing-ratio-etapa05.md) | IMPLEMENTED |
| [BL-08 — Incidente .env en historial público: rotar+purgar+privatizar](BL-08-incidente-env-historial.md) | PARTIAL |
| [BL-09 — Ledger doble FT-/AT- global (registries/ledger.jsonl)](BL-09-ledger-doble-ft-at.md) | IMPLEMENTED |
| [BL-10 — Backfill legacy_estimate de FT históricos (zoos)](BL-10-backfill-legacy-estimate-ft.md) | IMPLEMENTED |
| [BL-11 — Familias transversales de hipótesis (registries/families/)](BL-11-familias-transversales.md) | IMPLEMENTED |
| [BL-12 — Provenance FT→AT + enmienda constitución §2 (ADR)](BL-12-provenance-ft-at-adr.md) | IMPLEMENTED |
| [BL-13 — Campo surface en manifiestos/registry + normalize lo respeta](BL-13-campo-surface-manifiestos.md) | IMPLEMENTED |
| [BL-14 — Bloque components: receta congelada del predictor de v11](BL-14-components-passport-receta.md) | IMPLEMENTED |
| [BL-15 — Contrato forecast_output (Py+TS) + validación en el zoo](BL-15-contrato-forecast-output.md) | PARTIAL |
| [BL-16 — CI constitucional Etapa 0 (legalidad + serialización canónica)](BL-16-ci-constitucional-etapa0.md) | IMPLEMENTED |
| [BL-17 — Identidad: fingerprints + canonical writer + spine](BL-17-fingerprints-canonical-writer.md) | IMPLEMENTED |
| [BL-18 — Catálogo de métricas + motor único + metric_event](BL-18-catalogo-motor-metricas.md) | PARTIAL |
| [BL-19 — Esquema DB forecast. + rol forecast_writer](BL-19-schema-forecast-roles-db.md) | PARTIAL |
| [BL-20 — Vista admin SHAP/interpretabilidad por modelo×versión (ambas superficies)](BL-20-admin-shap-interpretabilidad.md) | IMPLEMENTED |
| [BL-21 — Event sourcing exec. (4 entornos) + idempotencia](BL-21-event-sourcing-exec.md) | PARTIAL |
| [BL-22 — fact_position / fact_pnl + identidad contable + timing_ratio persistido](BL-22-fact-position-pnl.md) | PARTIAL |
| [BL-23 — Backfill anti-supervivencia (campeonas+candidatas+retiradas+baselines)](BL-23-backfill-anti-supervivencia.md) | PARTIAL |
| [BL-24 — Linaje nodes/edges + camino dorado + revisiones tipificadas](BL-24-linaje-camino-dorado.md) | PARTIAL |
| [BL-25 — Monitoreo en tres relojes (control__system_health)](BL-25-monitoreo-tres-relojes.md) | PARTIAL |
| [BL-26 — portfolio_snapshot: barrera temporal del libro](BL-26-portfolio-snapshot.md) | PARTIAL |
| [BL-27 — Allocator v1 (inverse-vol+caps) + multiplicadores + gate de novedad](BL-27-allocator-v1-novedad.md) | PARTIAL |
| [BL-28 — Factories nuevas (data/strategy/forecast) + diff semántico](BL-28-factories-diff-semantico.md) | PARTIAL |
| [BL-29 — CLI qlab + cutoff impuesto por la capa de lectura](BL-29-qlab-cli-cutoff-lectura.md) | PARTIAL |
| [BL-30 — Servicio de ejecución fuera de Airflow + pre-trade + kill switch independiente](BL-30-execution-service-externo.md) | PARTIAL |
| [BL-31 — Migración strangler de USD/COP (L7 al final)](BL-31-strangler-cop.md) | PARTIAL |
| [BL-32 — Passport (vista live + MV) + Control Tower](BL-32-passport-control-tower.md) | PARTIAL |
| [BL-33 — Institutional Readiness Matrix con evidencias](BL-33-readiness-matrix.md) | PARTIAL |
| [BL-34 — Ruta /replay (alias de la sección de /dashboard)](BL-34-ruta-replay.md) | IMPLEMENTED |
| [BL-35 — URIs de datasets + arista prohibida forecast→allocator en parseo](BL-35-dataset-uris-arista-prohibida.md) | IMPLEMENTED |
| [BL-36 — Racionalización del inventario DB (59 tablas → matriz de verdad aplicada)](BL-36-racionalizacion-inventario-db.md) | PARTIAL |
| [BL-37 — Identidades canónicas (reference.asset/instrument/provider_symbol/bar_interval)](BL-37-identidades-canonicas.md) | PARTIAL |
| [BL-38 — Mercado canónico: raw_bar/canonical_bar + caggs 1h/4h/1d + política de resampleo](BL-38-market-canonical-resampleo.md) | PARTIAL |
| [BL-39 — Feature contracts por estrategia-versión + normalización al artefacto](BL-39-feature-contracts-normalizacion.md) | PARTIAL |
| [BL-40 — Calidad: cuarentena de anomalías + columnas fantasma](BL-40-calidad-cuarentena.md) | PARTIAL |
| [BL-41 — Seguridad DB P0: referencias externas, roles y timestamps](BL-41-seguridad-db-p0.md) | PARTIAL |
| [BL-42 — Unidades decimales + action.strategy_signal normalizada (JSONB de política)](BL-42-unidades-decimales-signal-normalizada.md) | PARTIAL |
| [BL-43 — Aislar el modelo sintético demo (CI que lo bloquee fuera de demo)](BL-43-demo-sintetica-aislada.md) | IMPLEMENTED |
| [BL-44 — TimescaleDB ops + perfil físico ampliado](BL-44-timescale-ops-perfil-fisico.md) | PARTIAL |
| [BL-45 — Motor de políticas: contrato + registry + factory (R1-R3)](BL-45-policy-engine-contrato.md) | PARTIAL |
| [BL-46 — Políticas: backend (policy_version/signal) + frontend schema-driven (R4-R5)](BL-46-policy-backend-frontend.md) | PARTIAL |
| [BL-47 — Migración de estrategias al motor de políticas (R6-R8)](BL-47-policy-migracion-r6-r8.md) | PARTIAL |
| [BL-48 — El costo de ejecución es la variable dominante en el intradía de USD/COP](BL-48-costos-ejecucion-intradia.md) | PLANNED |
| [BL-49 — Los dos tests de §13 que quedaron sin implementar](BL-49-tests-2-y-14-tesis.md) | PLANNED |
| [BL-50 — Reparación y re-evaluación de la tesis RL (EXP-TESIS-RL-01)](BL-50-reparacion-tesis-rl.md) | PARTIAL |

<!-- /idx -->
