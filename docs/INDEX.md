# USDCOP Trading System — Índice de documentación

> **Verificado contra disco el 2026-07-30.** La versión anterior de este índice databa de
> 2025-10-22 y apuntaba a un layout plano (`ARCHITECTURE.md`, `RUNBOOK.md`, … en la raíz de
> `docs/`) que dejó de existir cuando `docs/` se reorganizó en subdirectorios: **84 de sus
> enlaces estaban muertos**. Nadie lo notó porque el link-checker solo miraba `.claude/`.
> Ahora `docs/**` también entra en `specs-gate.yml`, así que este índice no puede volver a
> pudrirse en silencio.

**Dos árboles de conocimiento, con responsabilidades distintas:**

| Árbol | Qué contiene | Gobierno |
|---|---|---|
| [`.claude/`](../.claude/README.md) | Specs SDD, rules auto-cargadas, skills, agents | Front-matter tipado + `specs-gate.yml` |
| `docs/` (aquí) | Documentación de proyecto: arquitectura, runbooks, guías, ADRs, legal | Enlaces verificados en CI |

Si buscas **cómo está construido el sistema hoy**, empieza por
[`.claude/specs/architecture-overview.md`](../.claude/specs/architecture-overview.md).
Este árbol es documentación de acompañamiento, más narrativa y de proceso.

---

## Arranque

- [QUICK_START.md](guides/QUICK_START.md) — setup rápido de desarrollo
- [ONBOARDING_NEW_TEAM_MEMBER.md](guides/ONBOARDING_NEW_TEAM_MEMBER.md) — incorporación
- [DEVELOPMENT.md](guides/DEVELOPMENT.md) — entorno y estándares de código
- [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) — definición del proyecto
- [../README.md](../README.md) — overview del repositorio
- [../AGENTS.md](../AGENTS.md) — reglamento de agentes (Codex lo lee automáticamente)
- [../CLAUDE.md](../CLAUDE.md) — contexto de proyecto para Claude Code

## Arquitectura

- [ARCHITECTURE.md](architecture/ARCHITECTURE.md) — arquitectura completa del sistema
- [ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md) — diagramas Mermaid
- [ARCHITECTURE_CONTRACTS.md](architecture/ARCHITECTURE_CONTRACTS.md) — contratos entre capas
- [DATA_FLOW_ARCHITECTURE.md](architecture/DATA_FLOW_ARCHITECTURE.md) — flujo de datos
- [DATABASE_ER_DIAGRAM.md](architecture/DATABASE_ER_DIAGRAM.md) — modelo entidad-relación
- [INTEGRATION_MATRIX.md](architecture/INTEGRATION_MATRIX.md) — matriz de integraciones
- [SIGNALBRIDGE_SPEC_v2.md](architecture/SIGNALBRIDGE_SPEC_v2.md) — spec de SignalBridge
- [FORECASTING_PIPELINE_ANALYSIS.md](architecture/FORECASTING_PIPELINE_ANALYSIS.md) — pipeline de forecasting
- [MODEL_GOVERNANCE_POLICY.md](architecture/MODEL_GOVERNANCE_POLICY.md) — gobernanza de modelos
- [REPRODUCIBILITY.md](architecture/REPRODUCIBILITY.md) — reproducibilidad
- [DATA_VERSIONING.md](architecture/DATA_VERSIONING.md) — versionado de datos

### API

- [API_REFERENCE_V2.md](architecture/API_REFERENCE_V2.md) — RT Orchestrator + protocolo WebSocket
- [API_CONTRACTS_SHARED.md](architecture/API_CONTRACTS_SHARED.md) — contratos compartidos
- [API_ENDPOINTS_MULTIMODEL.md](architecture/API_ENDPOINTS_MULTIMODEL.md) — endpoints multi-modelo
- [API_VERSIONING.md](architecture/API_VERSIONING.md) — versionado de API

## Operaciones

- [RUNBOOK.md](operations/RUNBOOK.md) — procedimientos operativos
- [STARTUP_CHECKLIST.md](operations/STARTUP_CHECKLIST.md) — checklist de arranque
- [TROUBLESHOOTING.md](operations/TROUBLESHOOTING.md) — diagnóstico de problemas
- [INCIDENT_RESPONSE_PLAYBOOK.md](operations/INCIDENT_RESPONSE_PLAYBOOK.md) — respuesta a incidentes
- [DISASTER_RECOVERY_PLAYBOOK.md](operations/DISASTER_RECOVERY_PLAYBOOK.md) — recuperación ante desastres
- [DATABASE_ROLLBACK_RUNBOOK.md](operations/DATABASE_ROLLBACK_RUNBOOK.md) — rollback de base de datos
- [SYNC_RECOVERY_RUNBOOK.md](operations/SYNC_RECOVERY_RUNBOOK.md) — recuperación de sincronización
- [GAME_DAY_CHECKLIST.md](operations/GAME_DAY_CHECKLIST.md) — simulacro de fallos
- [SLA.md](operations/SLA.md) — acuerdos de nivel de servicio
- [PROMETHEUS_METRICS_REFERENCE.md](operations/PROMETHEUS_METRICS_REFERENCE.md) — referencia de métricas
- [EXPERIMENT_LAUNCH_CHECKLIST.md](operations/EXPERIMENT_LAUNCH_CHECKLIST.md) — lanzamiento de experimentos

### Calendario de trading y zona horaria

- [TIMEZONE_POLICY.md](operations/TIMEZONE_POLICY.md) — política de zona horaria
- [TRADING_CALENDAR_README.md](operations/TRADING_CALENDAR_README.md) — calendario de trading
- [TRADING_CALENDAR_INTEGRATION_EXAMPLES.md](operations/TRADING_CALENDAR_INTEGRATION_EXAMPLES.md) — ejemplos de integración
- [TRADING_CALENDAR_MIGRATION_CHECKLIST.md](operations/TRADING_CALENDAR_MIGRATION_CHECKLIST.md) — checklist de migración
- [TRADING_CALENDAR_VALIDATION.md](operations/TRADING_CALENDAR_VALIDATION.md) — validación
- [TRADING_CALENDAR_VALIDATION_REPORT.md](operations/TRADING_CALENDAR_VALIDATION_REPORT.md) — informe de validación

## Guías

- [DEPLOYMENT_GUIDE.md](guides/DEPLOYMENT_GUIDE.md) — despliegue
- [MIGRATION_GUIDE.md](guides/MIGRATION_GUIDE.md) — migración V1 → V2
- [MLOPS_PRODUCTION_GUIDE.md](guides/MLOPS_PRODUCTION_GUIDE.md) — MLOps en producción
- [MLFLOW_INTEGRATION_GUIDE.md](guides/MLFLOW_INTEGRATION_GUIDE.md) — integración con MLflow
- [DVC_INTEGRATION_GUIDE.md](guides/DVC_INTEGRATION_GUIDE.md) — integración con DVC
- [GRAFANA_DASHBOARDS_SETUP.md](guides/GRAFANA_DASHBOARDS_SETUP.md) — dashboards de Grafana
- [SSOT_USAGE_GUIDE.md](guides/SSOT_USAGE_GUIDE.md) — uso de los SSOT
- [AB_TESTING_GUIDE.md](guides/AB_TESTING_GUIDE.md) — A/B testing
- [AB_TESTING_END_TO_END_GUIDE.md](guides/AB_TESTING_END_TO_END_GUIDE.md) — A/B testing end-to-end
- [COST_MANAGEMENT_GUIDE.md](guides/COST_MANAGEMENT_GUIDE.md) — gestión de costos
- [PROJECT_REPLICATION_GUIDE.md](guides/PROJECT_REPLICATION_GUIDE.md) — replicación del proyecto

## Decisiones de arquitectura (ADR)

- [Índice de ADRs](adr/README.md) · [Plantilla](adr/TEMPLATE.md)
- [ADR-0001](adr/ADR-0001-wilder-ema-for-technical-indicators.md) — Wilder EMA para indicadores técnicos
- [ADR-0002](adr/ADR-0002-feature-circuit-breaker.md) — Circuit breaker de features
- [ADR-0003](adr/ADR-0003-redis-streams-for-realtime.md) — Redis Streams para tiempo real
- [ADR-0004](adr/ADR-0004-timescaledb-for-ohlcv.md) — TimescaleDB para OHLCV
- [ADR-0005](adr/ADR-0005-ppo-for-trading.md) — PPO para trading

> Los ADR más recientes del sistema SDD viven en
> [`.claude/specs/adr/`](../.claude/specs/adr/). Este directorio conserva los cinco originales.

## Seguridad y legal

- [SECURITY-env-leak-remediation.md](SECURITY-env-leak-remediation.md) — remediación de fuga de `.env`
- [rbac/](rbac/README.md) — documentación de control de acceso
- [legal/SFC-GATE-CHECKLIST.md](legal/SFC-GATE-CHECKLIST.md) — gate legal SFC Colombia

## Modelos y análisis

- [model_cards/](model_cards/README.md) — fichas de modelo
- [analysis/TRAINING_LINEAGE.md](analysis/TRAINING_LINEAGE.md) — linaje de entrenamiento
- [runbooks/](runbooks/README.md) — runbooks adicionales
- [templates/](templates/README.md) — plantillas
- [utils/README_BACKUP_UTILITIES.md](utils/README_BACKUP_UTILITIES.md) — utilidades de backup

## Material del curso

- [COURSE_PROJECT.md](COURSE_PROJECT.md) — proyecto de curso
- [defense_qa.md](defense_qa.md) — preguntas de defensa
- [slides/](slides/README.md) — presentaciones

---

## Cómo mantener este índice

1. Si añades un documento a `docs/`, **añádelo aquí**.
2. Si mueves o renombras uno, `python scripts/validation/check_knowledge_links.py`
   te dirá qué enlaces rompiste — córrelo antes de commitear.
3. **No escribas conteos ni tamaños de fichero en prosa.** La versión anterior de este índice
   anunciaba "ARCHITECTURE.md (50 KB)" para un fichero que ya no estaba ahí. Los números que
   describen el sistema salen de [`.claude/generated/inventory.json`](../.claude/generated/inventory.json).

## Documentos de este directorio

<!-- idx:auto -->

| Documento | Estado |
|---|---|
| [USDCOP Trading System — Final MLOps Project](COURSE_PROJECT.md) | — |
| [Preguntas de Defensa de Tesis --- Preparacion Completa](defense_qa.md) | — |
| [PROJECT_DEFINITION.md — GlobalMinds](PROJECT_DEFINITION.md) | — |
| [Remediación de fuga de .env — 2026-07-09](SECURITY-env-leak-remediation.md) | — |

**Subdirectorios:** [`adr/`](adr/README.md) · [`analysis/`](analysis/README.md) · [`architecture/`](architecture/README.md) · [`guides/`](guides/README.md) · [`legal/`](legal/SFC-GATE-CHECKLIST.md) · [`model_cards/`](model_cards/README.md) · [`operations/`](operations/README.md) · [`rbac/`](rbac/README.md) · [`runbooks/`](runbooks/README.md) · [`slides/`](slides/README.md) · [`templates/`](templates/README.md) · [`utils/`](utils/README_BACKUP_UTILITIES.md)

<!-- /idx -->
