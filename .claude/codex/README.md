---
kind: audit
status: IMPLEMENTED
version: 2.0.0
last_verified: 2026-07-30
supersedes: []
code_anchors:
  - README.md
  - CLAUDE.md
  - AGENTS.md
  - .claude/specs/audit/AUDIT-2026-07-remediation.md
  - .claude/specs/audit/STRATEGIC-ASSESSMENT-2026-07.md
  - .claude/specs/platform/codex-review-integration.md
---

# Codex — canal de revisión independiente

Esta carpeta conserva las revisiones independientes de Codex sobre el sistema: diferencias
entre lo documentado, lo implementado y lo demostrado con pruebas. **No reemplaza los SSOT**
de [`../rules/`](../rules/00-INDEX.md) ni [`../specs/`](../specs/README.md).

- **Cómo se opera Codex** (perfiles de sandbox, flags, validación de config):
  [`../specs/platform/codex-review-integration.md`](../specs/platform/codex-review-integration.md)
- **Protocolo de mensajería Claude↔Codex** (inbox, leases, ACK):
  [`../coordination/PROTOCOL.md`](../coordination/PROTOCOL.md)
- **Reglamento que Codex lee al arrancar**: [`../../AGENTS.md`](../../AGENTS.md)

---

## Estructura (reorganizada 2026-07-30)

Hasta 2026-07-30 esto era un **volcado plano de 71 ficheros** en un solo directorio, mezclando
informes, planes, propuestas de spec, logs y scratch de proceso; **45 de ellos no pasaban el
gate de front matter**. Se reubicó todo (sin borrar nada) en una taxonomía explícita. El
mapeo completo origen→destino está en `MOVE-MANIFEST.json`.

| Directorio | Contiene | `kind` |
|---|---|---|
| `audits/` | Hallazgos puntuales: auditorías, revisiones adversariales, verificaciones, pentest | `audit` |
| `plans/` | Planes de remediación/desbloqueo y el backlog consolidado | `roadmap` |
| `proposals/` | Specs propuestas por Codex, **aún no promovidas** a `../specs/` | `roadmap` |
| `logs/` | Logs de ejecución y estado de harness (punto en el tiempo) | `audit` |
| `inventories/` | Inventarios y matrices de cobertura | `audit` |
| `governance/` | Gobernanza durable: trazabilidad, skills, ciclo de checkpoints | `audit` |
| `evidence/` | Artefactos probatorios (bundles, capturas, salidas) | — |
| `harness/` | Ejecutor reproducible de gates y sus contratos | — |
| `_runtime/` | **Gitignorado.** Scratch de una sola corrida: manifiestos pytest, deltas de monitor, XML/JSON | — |

### Puertas de entrada

- [`audits/SPEC-FINDINGS-2026-07-20.md`](audits/SPEC-FINDINGS-2026-07-20.md) — auditoría inicial de plataforma, frontend, backend, trading y forecasting
- [`plans/BACKLOG.md`](plans/BACKLOG.md) — backlog consolidado con criterios de cierre
- [`governance/TRACEABILITY-MATRIX.md`](governance/TRACEABILITY-MATRIX.md) — requisito → riesgo → prueba → evidencia
- [`governance/SKILLS-GOVERNANCE.md`](governance/SKILLS-GOVERNANCE.md) — procedencia y límites de skills externas
- [`harness/run-quality-gates.ps1`](harness/run-quality-gates.ps1) — ejecutor reproducible de gates

---

## Reglas de trabajo

1. Cada afirmación apunta a una spec, archivo de código, configuración, prueba o resultado medido.
2. Un resultado de backtest **no** se describe como evidencia de rentabilidad futura
   (ver [`../rules/quant-constitution.md`](../rules/quant-constitution.md)).
3. Los hallazgos no modifican por sí solos el contrato oficial: incorporarlos exige actualizar
   el SSOT correspondiente **y sus pruebas**.
4. Todo hallazgo indica severidad, impacto, recomendación y estado.
5. **Todo documento nuevo aquí lleva front matter tipado.** Los `kind` válidos son los del
   contrato (`rule`, `as-built`, `roadmap`, `adr`, `audit`, `historical`) — no se inventan
   nuevos. En la reorganización hubo que traducir cinco kinds improvisados
   (`specification`, `implementation-log`, `evidence`, `strategy-plan`, `release-plan`) y
   cinco status fuera de contrato; un vocabulario que crece cada vez que un agente improvisa
   deja de ser un contrato.
6. **El scratch de proceso va a `_runtime/`**, nunca a la raíz de esta carpeta.

## Documentos de este directorio

<!-- idx:auto -->

**Subdirectorios:** [`audits/`](audits/README.md) · [`governance/`](governance/README.md) · [`harness/`](harness/README.md) · [`inventories/`](inventories/README.md) · [`logs/`](logs/README.md) · [`plans/`](plans/README.md) · [`proposals/`](proposals/README.md)

<!-- /idx -->
