---
kind: audit
status: PARTIAL
version: 1.2.0
last_verified: 2026-07-31
supersedes: []
code_anchors:
  - tests/regression/test_readiness_matrix.py
  - tests/regression/test_contract_mirrors.py
  - services/signalbridge_api/app/services/pretrade.py
  - scripts/validation/check_trial_ledger.py
  - usdcop-trading-dashboard/lib/contracts/rbac.contract.ts
---

# Institutional readiness matrix

> **Responsibility:** registrar qué control institucional se espera, qué evidencia existe hoy y
> qué falta, sin convertir presencia de código o documentación en una afirmación de readiness.
>
> Fuentes: [evaluación institucional](03-institutional-readiness.md),
> [BL-33](backlog/BL-33-readiness-matrix.md),
> [constitución quant](../../rules/quant-constitution.md) y
> [protocolo de aprobación](../../rules/approval-gates.md).

Esta matriz evalúa el repositorio y la evidencia operativa disponible. `VERIFIED_REPO` no equivale
a control probado en producción, certificación, auditoría independiente ni autorización legal.
La matriz **no autoriza capital**, no promueve modelos y no convierte el sistema en el Caso B de
administración de dinero de terceros.

La existencia de un enlace no prueba correspondencia. El gate mantiene un target-set revisado para
cada `Control ID`: sustituir la evidencia de una fila por otro archivo existente —por ejemplo
`LICENSE`— falla aunque el enlace resuelva. Un cambio legítimo de evidencia modifica registro y pin
en el mismo review; esto evita convertir “archivo presente” en “afirmación demostrada”.

## Semántica de estado

| State | Meaning |
|---|---|
| `VERIFIED_REPO` | El control tiene un oráculo ejecutable en el repo y pasó en el corte indicado; no prueba operación real. |
| `PARTIAL` | Existe implementación o evidencia útil, pero falta alcance, integración, simulacro o revisión independiente. |
| `BLOCKED_EXTERNAL` | El siguiente paso exige autoridad, proveedor, identidad o infraestructura que el agente no puede crear. |
| `NOT_EVIDENCED` | No hay artefacto verificable suficiente; un documento de intención no cuenta como implementación. |

## Registro de controles

| Control ID | Domain | Control | Expected evidence | Observed evidence | State | Owner | Verified |
|---|---|---|---|---|---|---|---|
| SEC-01 | Security | Credenciales expuestas e historial público saneado | Revocación comprobada con cada proveedor, historial sin `.env`, decisión de visibilidad y escaneo posterior | [BL-08](backlog/BL-08-incidente-env-historial.md) y [runbook de remediación](../../../docs/SECURITY-env-leak-remediation.md) documentan un incidente abierto; no se leen secretos para validarlo | BLOCKED_EXTERNAL | Operator + CODEX / BL-08 | 2026-07-31 |
| SEC-02 | Security | Segregación de identidades y cuatro ojos | Identidades distintas para research, risk, execution y administración; aprobación crítica por otro principal y audit trail | [Plan institucional](03-institutional-readiness.md) verifica concentración en una sola identidad; el protocolo dual no sustituye segregación humana | BLOCKED_EXTERNAL | Operator / governance | 2026-07-31 |
| TECH-01 | Technology | Contratos compartidos Python/TypeScript en paridad | Espejos actualizados juntos y gate que falla si forma, enums o campos divergen | [Gate de contract mirrors](../../../tests/regression/test_contract_mirrors.py) es ejecutable y forma parte del corte BL-33 | VERIFIED_REPO | CODEX / CI | 2026-07-31 |
| TECH-02 | Technology | Identidad canónica de decisiones y artefactos | Fingerprint determinista desde inputs completos, writer único y round-trip persistente | [BL-17](backlog/BL-17-fingerprints-canonical-writer.md) conserva gaps de writer único e integración; no se eleva por tests aislados | PARTIAL | CODEX / BL-17 | 2026-07-31 |
| TECH-03 | Technology | Datos PIT, frescura y linaje hasta snapshot | Cutoff obligatorio, disponibilidad causal, mercado canónico y lineage raw-to-decision | [BL-24](backlog/BL-24-linaje-camino-dorado.md), [BL-29](backlog/BL-29-qlab-cli-cutoff-lectura.md) y [BL-38](backlog/BL-38-market-canonical-resampleo.md) declaran brechas complementarias | PARTIAL | CODEX / BL-24,29,38 | 2026-07-31 |
| TECH-04 | Technology | Backup y restore reproducibles | Restore ejecutado desde backup sellado, secuencias coherentes, RTO/RPO medidos y copia fuera de región | [Guard de secuencias](../../../tests/regression/test_restore_resyncs_sequences.py) cubre una frontera; el [playbook DR](../../../docs/operations/DISASTER_RECOVERY_PLAYBOOK.md) no demuestra failover cross-region | PARTIAL | Operations + CODEX | 2026-07-31 |
| TECH-05 | Technology | Salud y observabilidad del sistema | Liveness/readiness, métricas, tres relojes y alertas con fallo observable | [BL-25](backlog/BL-25-monitoreo-tres-relojes.md) tiene control parcial y gaps de cableado productivo | PARTIAL | CLAUDE / BL-25 | 2026-07-31 |
| TECH-06 | Technology | Plan de migraciones sellado y revisado | Cada plan coincide con su digest pinneado, el cambio exige review y el runner falla ante drift | La [prueba de contratos de seguridad](../../../tests/unit/test_codex_safety_contracts.py) detectó que `fabric-v1` difiere hoy de su digest revisado | PARTIAL | CODEX / migration governance | 2026-07-31 |
| RISK-01 | Risk | Pre-trade risk bloqueante y paper-first | Excepción o dependencia caída bloquea; PAPER no alcanza broker; límites forman parte de identidad | [Contrato de ejecución](../platform/execution-bridge.md) y [pruebas de seguridad](../../../tests/unit/test_codex_safety_contracts.py) contienen oráculos ejecutables | VERIFIED_REPO | Shared / execution risk | 2026-07-31 |
| RISK-02 | Risk | Kill switch independiente y auditable | Activación bloquea trading, reset exige confirmación, acción sobrevive caída de Airflow y queda en ledger | [Trading flags](../../../tests/regression/test_trading_flags.py) y [command tests](../../../tests/unit/test_command_pattern.py) cubren lógica; [BL-31](backlog/BL-31-strangler-cop.md) mantiene el simulacro independiente sin atestar | PARTIAL | Shared / risk operations | 2026-07-31 |
| RISK-03 | Risk | Disciplina anti-selección y cobro de trials | Ledger encadenado, familias completas, conteos DSR reconciliados y ningún juez reanclado | [Gate del trial ledger](../../../tests/regression/test_trial_ledger.py) y [constitución quant](../../rules/quant-constitution.md) fijan el control del repositorio | VERIFIED_REPO | Shared / quant governance | 2026-07-31 |
| RISK-04 | Risk | Snapshot de portfolio y allocator fail-closed | Partición causal de señales, políticas coherentes, optimización acotada y fallback seguro | [BL-26](backlog/BL-26-portfolio-snapshot.md) y [BL-27](backlog/BL-27-allocator-v1-novedad.md) registran gaps de identidad, política y persistencia | PARTIAL | CODEX / BL-26,27 | 2026-07-31 |
| RISK-05 | Risk | Riesgo de contraparte, liquidez y capacidad | Exposición por broker, cash/margin, impacto, concentración, límites de volumen y stress verificable | [Plan institucional](03-institutional-readiness.md) define el control; no existe cobertura operativa integral en el corte | NOT_EVIDENCED | Portfolio + risk | 2026-07-31 |
| RISK-06 | Risk | API e identidad del motor de métricas | Constructor, catálogo, annualización e identidad permanecen alineados y los consumidores instancian el contrato vigente | [BL-18](backlog/BL-18-catalogo-motor-metricas.md) sigue PARTIAL y [su prueba de seguridad](../../../tests/unit/test_codex_safety_contracts.py) falla hoy porque aún pasa `annualization_by_asset` a una API que ya no lo acepta | PARTIAL | CODEX / BL-18 | 2026-07-31 |
| EXEC-01 | Execution | Idempotencia, fencing y compare-and-swap | Reintento conserva identidad, lease impide doble dispatcher y transición obsoleta falla cerrada | [Pruebas de seguridad](../../../tests/unit/test_codex_safety_contracts.py) cubren fronteras; [BL-30](backlog/BL-30-execution-service-externo.md) mantiene el servicio externo incompleto | PARTIAL | CODEX / BL-30 | 2026-07-31 |
| EXEC-02 | Execution | Órdenes event-sourced y reconciliación firmada | Comandos/eventos inmutables, fills y posiciones conciliados, correcciones enlazadas y cierre EOD | [BL-21](backlog/BL-21-event-sourcing-exec.md) y [BL-22](backlog/BL-22-fact-position-pnl.md) siguen PARTIAL por persistencia e integración real | PARTIAL | CODEX / BL-21,22 | 2026-07-31 |
| EXEC-03 | Execution | Doble voto y revalidación server-side | Voto humano separado, revalidación al ejecutar, RBAC y rechazo de estado stale | [Approval gates](../../rules/approval-gates.md) y [gate de store privado](../../../tests/regression/test_approval_store_private.py) prueban piezas; SEC-02 impide afirmar cuatro ojos institucional | PARTIAL | CLAUDE / governance | 2026-07-31 |
| EXEC-04 | Execution | Paridad replay/paper/live y migración por capas | Semantic hashes comparables, ventanas prospectivas y rollback antes de retirar el camino anterior | [BL-31](backlog/BL-31-strangler-cop.md) conserva capas no atestadas y ventanas pendientes | PARTIAL | CLAUDE / BL-31 | 2026-07-31 |
| EXEC-05 | Execution | Retiro EXIT_ONLY y liquidación segura | Retiro impide nuevas entradas, conserva gestión de salida y prueba broker/mercado cerrado | [Withdrawal protocol](../assets/usdcop/WITHDRAWAL-PROTOCOL.md) documenta el flujo; no hay atestación integral live | PARTIAL | Risk + execution | 2026-07-31 |
| SEC-03 | Security | RBAC deny-by-default y rol distinto de plan | Toda ruta protegida aparece en contrato, privilegios efectivos se prueban y rol no compra capacidad | [Regla RBAC](../../rules/rbac.md), [contrato](../../../usdcop-trading-dashboard/lib/contracts/rbac.contract.ts) y [coverage gate](../../../usdcop-trading-dashboard/scripts/check-rbac-coverage.mjs) forman el oráculo del repo | VERIFIED_REPO | CLAUDE / RBAC | 2026-07-31 |
| SEC-04 | Security | Vault real y segregación de credenciales DB | Secret manager externo, referencias sin secreto, roles no-superusuario, rotación y deny desde frontend/Airflow genérico | [BL-41](backlog/BL-41-seguridad-db-p0.md) está PLANNED y prohíbe DDL/cutover sin Vault y roles reales | BLOCKED_EXTERNAL | Operator + CODEX / BL-41 | 2026-07-31 |
| SEC-05 | Security | MFA, rotación programada, segmentación y gestión de proveedores | MFA obligatorio, rotación ensayada, redes separadas, SBOM/vuln SLA y evaluación de terceros | [Spec de autenticación](../platform/authentication.md) enumera gaps; no hay evidencia integral de estos controles | NOT_EVIDENCED | Security + Operator | 2026-07-31 |
| SEC-06 | Security | Respuesta a incidentes ejercitada | Roles, severidad, contención, preservación de evidencia, comunicaciones y postmortem ensayados | Existe [playbook de incidentes](../../../docs/operations/INCIDENT_RESPONSE_PLAYBOOK.md), pero no un simulacro independiente sellado | PARTIAL | Security + Operations | 2026-07-31 |
| COMP-01 | Compliance | Provenance y conservación de registros de investigación | Trials y decisiones append-only, fuentes resolubles, hash-chain y correcciones sin reescribir historia | [Gobernanza BL-09/11/12](../../../tests/regression/test_bl09_bl11_bl12_governance.py) y [trial ledger](../../../tests/regression/test_trial_ledger.py) son oráculos ejecutables | VERIFIED_REPO | Shared / quant governance | 2026-07-31 |
| COMP-02 | Compliance | Gate legal antes de capital de terceros | Opinión legal/jurisdicción, vehículo y permisos documentados; modo real bloqueado hasta aprobación | [Regla RBAC](../../rules/rbac.md) exige gate SFC y paper-only; no existe determinación legal externa para Caso B | BLOCKED_EXTERNAL | Operator + legal counsel | 2026-07-31 |
| COMP-03 | Compliance | KYC/AML, conflictos, ética y operaciones personales | Políticas aprobadas, responsables, evidencias de screening, excepciones y revisión periódica | [Plan institucional](03-institutional-readiness.md) identifica el paquete como ausente; software no lo sustituye | NOT_EVIDENCED | Compliance + Operator | 2026-07-31 |
| COMP-04 | Compliance | Revisión independiente y retención regulatoria | Calendario, muestras, sign-off humano independiente, legal hold y registro de excepciones | [Plan institucional](03-institutional-readiness.md) exige revisión independiente; no hay artefacto de ejecución humana | NOT_EVIDENCED | Compliance | 2026-07-31 |
| OPS-01 | Operations | DR con RTO/RPO y failover fuera de región | Objetivos aprobados, restore cronometrado, pérdida medida y failover cross-region repetible | El [playbook DR](../../../docs/operations/DISASTER_RECOVERY_PLAYBOOK.md) es una guía; no hay evidencia cross-region ni RTO/RPO sellados | NOT_EVIDENCED | Operations + Operator | 2026-07-31 |
| OPS-02 | Operations | Runbooks e incidentes ejecutados por otra persona | Ejecución fechada por operador distinto, artefactos, tiempos, desviaciones y acciones cerradas | [Playbook de incidentes](../../../docs/operations/INCIDENT_RESPONSE_PLAYBOOK.md) existe; [plan institucional](03-institutional-readiness.md) registra que nunca fue ejecutado por un tercero | PARTIAL | Operations | 2026-07-31 |
| OPS-03 | Operations | Reconciliaciones diarias con firma independiente | Conciliación pre-open/EOD, diferencias resueltas y firma de persona distinta al trader | [BL-22](backlog/BL-22-fact-position-pnl.md) sólo cubre parte del dato operativo; no hay sign-off humano independiente | NOT_EVIDENCED | Operations + accounting | 2026-07-31 |
| OPS-04 | Operations | Monitoreo de datos, modelo y PnL con escalamiento | Tres relojes, frescura fail-closed, alarmas accionables, on-call y evidencia de resolución | [BL-25](backlog/BL-25-monitoreo-tres-relojes.md) entrega candados parciales, no el ciclo operativo completo | PARTIAL | CLAUDE / BL-25 | 2026-07-31 |
| OPS-05 | Operations | Pruebas de pérdida de proveedor, broker y plataforma | Inyección de caída, fallback permitido, bloqueo cuando no lo hay y recuperación registrada | [Plan institucional](03-institutional-readiness.md) exige las pruebas; no hay campaña integral sellada | NOT_EVIDENCED | Operations + execution | 2026-07-31 |
| INV-01 | Investors | Superficies honestas y datos sintéticos aislados | Toda cifra declara origen, demo no llega a dominio real y fallos upstream no fabrican performance | [Gate sintético](../../../tests/unit/test_codex_phase2_backlog.py), [test de API](../../../usdcop-trading-dashboard/tests/unit/api/synthetic-backtest-honesty.test.ts) y [BL-43](backlog/BL-43-demo-sintetica-aislada.md) muestran avance con alcance pendiente | PARTIAL | Shared / BL-43 | 2026-07-31 |
| INV-02 | Investors | NAV y valoración independientes | Libro contable, precios independientes, cash/fees/accruals, custodio y reconciliación con PnL operativo | [Plan institucional](03-institutional-readiness.md) confirma que `fact_pnl` no es NAV legal y que la capacidad falta | NOT_EVIDENCED | Accounting + administrator | 2026-07-31 |
| INV-03 | Investors | Suscripciones, redenciones, fees y high-water mark | Registro de inversionistas, clases, liquidez, fees reproducibles, notices y controles de redención | [Plan institucional](03-institutional-readiness.md) ubica estas funciones fuera del Quant Control Plane y sin implementación | NOT_EVIDENCED | Fund administration | 2026-07-31 |
| INV-04 | Investors | Reporting y performance auditados | Reportes periódicos, metodología, drawdowns, auditor independiente y trazabilidad hasta custodio/NAV | [Plan institucional](03-institutional-readiness.md) exige auditoría y reporting externo; el repo sólo contiene evidencia técnica propia | NOT_EVIDENCED | Investor relations + auditor | 2026-07-31 |

## Lectura del corte

- Los controles `VERIFIED_REPO` tienen oráculos ejecutables, pero siguen limitados al árbol local.
- Los controles `PARTIAL` no se promueven por presencia de archivos o por una suite focal.
- Los `BLOCKED_EXTERNAL` requieren una acción del operador, proveedor, legal counsel o
  infraestructura real; un agente no puede simular esa autoridad.
- Los `NOT_EVIDENCED` permanecen visibles para impedir que el término “institutional-grade” los
  oculte.

## Verificación reproducible

El corte se verifica con:

```powershell
python -m pytest tests/regression/test_readiness_matrix.py -q
python -m pytest tests/regression/test_contract_mirrors.py tests/unit/test_codex_safety_contracts.py tests/regression/test_trading_flags.py tests/unit/test_command_pattern.py tests/regression/test_trial_ledger.py tests/regression/test_bl09_bl11_bl12_governance.py tests/regression/test_restore_resyncs_sequences.py tests/regression/test_approval_store_private.py -q
node usdcop-trading-dashboard/scripts/test-rbac-contract.mjs
node usdcop-trading-dashboard/scripts/check-rbac-coverage.mjs
python scripts/validation/check_knowledge_links.py
```

Los resultados reales pertenecen al review pack de BL-33. Un comando listado pero no ejecutado no
cuenta como verde.

### Resultado de este corte

- Gate propio BL-33: `3 failed` contra la matriz decorativa original y `3 passed` después del
  registro verificable.
- Cross-review `CLD-267`: reemplazar `INV-04` por un enlace existente a `LICENSE` dejó el gate
  anterior verde. El R2 pinnea los targets revisados de todas las filas; gate ampliado `5 passed` y
  la misma sustitución ahora produce un error de correspondencia sobre `INV-04`.
- Batería factual amplia (segunda invocación): `199 passed, 3 failed, 1 skipped`; sumada al gate
  propio anterior, el agregado es `202 passed, 3 failed, 1 skipped`. Los fallos no se atribuyen a
  BL-33: dos exponen deriva del constructor de `MetricEngine` y uno expone drift del digest
  `fabric-v1`. Quedan visibles en `RISK-06` y `TECH-06`.
- Focal exacto de pre-trade, idempotencia, broker timeout, kill actions y fencing: `7 passed`.
- Contrato RBAC y cobertura RBAC: verdes. Frontmatter: `992 passed`. Enlaces relativos:
  `664 internal links resolve`.

## Cross-references

| Concern | Source |
|---|---|
| Criterio institucional y límite Caso A/Caso B | [Institutional readiness](03-institutional-readiness.md) |
| Backlog y aceptación | [BL-33](backlog/BL-33-readiness-matrix.md) |
| Arquitectura as-built | [Architecture overview](../architecture-overview.md) |
| Seguridad DB pendiente | [BL-41](backlog/BL-41-seguridad-db-p0.md) |
| Incidente de secretos | [BL-08](backlog/BL-08-incidente-env-historial.md) |

## DO NOT

- No convertir `VERIFIED_REPO` en evidencia live, auditoría independiente o autorización legal.
- No cerrar una fila enlazando sólo una intención; el estado debe reflejar la evidencia observada.
- No bajar un estado por conveniencia ni subirlo sin comando, simulacro o artefacto verificable.
- No escribir secretos, métricas de performance sin fuente ni conteos arquitectónicos manuales.
