---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - README.md
  - CLAUDE.md
  - .claude/specs/README.md
  - .claude/specs/audit/AUDIT-2026-07-remediation.md
  - .claude/specs/audit/STRATEGIC-ASSESSMENT-2026-07.md
  - usdcop-trading-dashboard/package.json
---

# Hallazgos de specs — 2026-07-20

## Alcance

Revisión inicial de las funcionalidades declaradas, arquitectura, backend, frontend, estrategia de trading,
forecasting y controles metodológicos. Este documento registra diferencias observadas; no sustituye las
specs autoritativas.

## Resumen ejecutivo

La plataforma posee una base sólida de contratos, gobierno metodológico, MLOps y controles de riesgo. Sin
embargo, la complejidad de infraestructura y la presentación del producto están por delante de la evidencia
de alpha. El comportamiento favorable observado proviene principalmente del control de riesgo —gate de
régimen, dimensionamiento por volatilidad y salidas—, no de una capacidad predictiva demostrada.

## Hallazgos

| ID | Severidad | Área | Hallazgo | Evidencia principal | Recomendación | Estado |
|---|---|---|---|---|---|---|
| CX-001 | HIGH | Specs | El README público declara H1 como producción y H5 como paper, mientras la spec vigente declara H5 producción y H1 pausado. | `README.md` vs `CLAUDE.md` | Corregir el README y generar estados desde un inventario SSOT. | OPEN |
| CX-002 | HIGH | Producto | Conteos de páginas, workflows, DAGs, migraciones y servicios aparecen duplicados y pueden quedar obsoletos. | `README.md`, `CLAUDE.md`, inventario generado | Sustituir conteos manuales por bloques generados y validados en CI. | OPEN |
| CX-003 | CRITICAL | Trading | Los titulares de rentabilidad pueden interpretarse como edge probado, pero USD/COP no supera el ajuste por selección y 2026 tiene una muestra mínima. | `STRATEGIC-ASSESSMENT-2026-07.md` | Describir el sistema como plataforma de investigación y control de riesgo hasta superar forward/OOS limpio. | OPEN |
| CX-004 | HIGH | Forecasting | Ridge/BR presentan R² OOS negativo y el model zoo tiene precisión direccional cercana a azar. | `STRATEGIC-ASSESSMENT-2026-07.md` | Separar claramente forecasting diagnóstico de señales autorizadas para ejecución. | OPEN |
| CX-005 | HIGH | Frontend | El quality gate de ESLint no está verde: 313 errores y 2397 warnings en la revisión del 2026-07-20. | `npm run lint` | Resolver primero errores de parseo, después código productivo, contratos generados y tests. | OPEN |
| CX-006 | HIGH | Arquitectura | El frontend es principalmente file-driven, mientras el backend/DB se presenta como fuente operativa; existen dos verdades parciales. | `public/data/**`, rutas BFF, specs de arquitectura | Elegir una fuente canónica o etiquetar cada respuesta como live, batch, replay, cache o synthetic. | OPEN |
| CX-007 | HIGH | Backend | Persisten TODOs de seguridad/operación: validación de webhook, autenticación WS, ownership, password reset, cierre en broker y auditoría de ejecución. | `services/signalbridge_api/**`, `lib/auth/**`, `services/inference_api/**` | Convertirlos en gates explícitos que bloqueen la declaración `LIVE_READY`. | OPEN |
| CX-008 | MEDIUM | Infraestructura | La cantidad de servicios y DAGs supera el uso efectivo; varios componentes están pausados, manuales o parcialmente conectados. | specs de observabilidad y evaluación estratégica | Retirar o archivar componentes sin consumidor y verificar capacidades end-to-end. | OPEN |
| CX-009 | HIGH | BTC | El modelo price-only es insuficiente para sostener una afirmación predictiva robusta. | estrategia BTC y evaluación OOS | Priorizar funding, OI, basis, flujos y on-chain antes de ampliar el model zoo. | OPEN |
| CX-010 | HIGH | Validación | Los backtests deben modelar selección múltiple, costes, latencia, rechazo de órdenes, embargo y regímenes. | constitución cuantitativa y tests de selection bias | Hacer obligatorios DSR > 0.95, OOS positivo y forward limpio para promoción. | PARTIAL |

## Verificación ejecutada

- Suite dirigida de specs, contratos, seguridad y metodología cuantitativa: **470 passed, 5 skipped,
  0 failed**.
- Frontend `npm run lint`: **313 errors, 2397 warnings**.
- Los skips correspondieron a disponibilidad del loader/dataset y casos donde la regla de muestra pequeña
  no aplicaba; no se interpretan como validación de rentabilidad.

## Orden recomendado

1. Corregir la verdad pública del producto y los estados de cada track.
2. Llevar lint, typecheck y build del frontend a verde y hacerlos gates de CI.
3. Definir la fuente canónica frontend/backend y añadir metadatos de procedencia/frescura.
4. Cerrar gaps de seguridad y ejecución antes de habilitar trading no-paper.
5. Evaluar estrategias simples y model-free contra el model zoo bajo el mismo protocolo.
6. Reducir servicios y DAGs que no aporten evidencia, seguridad u operación verificable.

## Criterio de cierre

Un hallazgo solo pasa a `DONE` cuando la spec SSOT, la implementación y una prueba automatizada o evidencia
operativa verificable coinciden. Una corrección únicamente documental se marca `DOC_FIXED`, no `DONE`.

