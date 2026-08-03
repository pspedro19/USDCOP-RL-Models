# AGENTS.md — USDCOP Trading System

> Reglamento de agentes para este repositorio. **Codex CLI lee este archivo automáticamente.**
> Claude Code lee `CLAUDE.md`, que gobierna lo mismo desde el otro lado.
>
> **Este archivo NO es una fuente de verdad paralela.** Es un índice de arranque + las
> prohibiciones duras. Toda regla sustantiva vive en `.claude/rules/*.md` y toda referencia
> densa en `.claude/specs/**`. Si algo aquí contradice una rule, **gana la rule**.

---

## 1. Lee esto antes de tocar nada

| Necesitas | Lee |
|---|---|
| Las reglas siempre-verdaderas (datos, frescura, estrategias, aprobación, RBAC, experimentos) | `.claude/rules/00-INDEX.md` → y de ahí cada `rules/*.md` |
| **Disciplina anti-selección — manda sobre código, specs y opiniones** | `.claude/rules/quant-constitution.md` |
| Arquitectura como está construida | `.claude/specs/architecture-overview.md` |
| Mapa del knowledge base y sus convenciones | `.claude/README.md` |
| Conteos reales (DAGs, páginas, rutas API, tablas) | `.claude/generated/inventory.json` — **generado, nunca a mano** |

Las `rules/` son cortas a propósito (hay presupuesto de palabras verificado en CI). Léelas
enteras; no las resumas de memoria.

---

## 2. Estructura

```
.claude/          Knowledge base gobernado por CI (specs, rules, skills, agents, templates)
  rules/          Reglas auto-cargadas — thin, siempre verdaderas
  specs/          Referencia densa on-demand
  skills/         Flujos operativos ejecutables
  agents/         Revisores especializados (read-only)
  coordination/   Estado runtime del protocolo dual Claude↔Codex (§5)
  generated/      Derivado del código — NO editar
docs/             Documentación de proyecto (ADRs, runbooks, guías)
src/              Python: forecasting, contracts, execution, risk, news_engine, analysis
services/         SignalBridge API y servicios comunes
airflow/dags/     Orquestación
scripts/          Entrypoints por propósito — NUNCA en la raíz de scripts/ (test lo enforce)
usdcop-trading-dashboard/   Next.js 15 App Router
config/           SSOT YAML (pipeline, macro, ejecución, activos)
tests/regression/ Gates de conocimiento y de layout
```

---

## 3. Comandos

```bash
make test            # suite completa
make lint            # ruff
make typecheck       # mypy
make check           # lint + typecheck + validate
make validate        # SSOT + contratos
```

Gate de conocimiento (lo que corre `specs-gate.yml` — córrelo si tocas `.claude/**` o `docs/**`):

```bash
python scripts/diagnostics/generate_inventory.py --check     # nunca --write en un gate
python scripts/diagnostics/generate_doc_indexes.py --check   # nunca --write en un gate
python -m pytest tests/regression/test_knowledge_frontmatter.py -q
python -m pytest tests/regression/test_knowledge_inventory.py -q
python -m pytest tests/regression/test_knowledge_autoload_budget.py -q
python -m pytest tests/regression/test_scripts_layout.py -q
python -m pytest tests/regression/test_contract_mirrors.py -q
python scripts/validation/check_knowledge_links.py
python -m pytest tests/regression/test_knowledge_links.py -q
python scripts/validation/check_knowledge_graph.py
python -m pytest tests/regression/test_knowledge_graph.py -q
```

---

## 4. Prohibiciones duras

**Secretos y datos**
- NUNCA leas, imprimas, cites ni copies `.env`, `.env.*`, `secrets/`, `*.pem`, `*.key`,
  `credentials*.json`, `service-account*.json`. Este repo tiene antecedente de fuga pública.
- NUNCA hardcodees claves de exchange — van por Vault.
- NUNCA edites una migración ya aplicada — crea una nueva.

**Conocimiento**
- NUNCA edites `.claude/generated/**` a mano — sale de `generate_inventory.py`.
- NUNCA escribas un conteo arquitectónico en prosa (DAGs, páginas, rutas). Si necesitas un
  número, sale del inventario generado. Un test lo verifica.
- Todo documento nuevo bajo `.claude/**` lleva front-matter tipado
  (`kind`, `status`, `version`, `last_verified`, `supersedes`, `code_anchors`). Ver
  `.claude/templates/spec-template.md`.
- **Enlaces: solo markdown relativo** — la forma `[texto](ruta.md)`. Los `[[wikilinks]]` no los verifica el
  link-checker, así que crean rot silencioso. Obsidian resuelve los relativos igual de bien.
- Una regla nueva va a `specs/`, no a `rules/`, salvo que sea siempre-verdadera — `rules/`
  entra en cada sesión y tiene presupuesto.

**Contratos**
- Un contrato cambia en **ambos** espejos a la vez (Python en `src/contracts/` o
  `services/**/contracts/`, TypeScript en `usdcop-trading-dashboard/lib/contracts/`).
  `test_contract_mirrors.py` conoce el mapa.
- Ninguna ruta nueva de `/api/**` sin entrada en `rbac.contract.ts`.

**Quant** — lee `.claude/rules/quant-constitution.md` completa; en resumen operativo:
- Ningún claim de edge sin Deflated Sharpe recomputado con el conteo de trials actualizado.
- Ningún parámetro elegido por grid-search sobre el test/OOS.
- Nada de Sharpe ni p-value con N < 20 trades.

---

## 5. Protocolo dual Claude ↔ Codex

Este repo opera con dos agentes en paralelo. El estado vive en `.claude/coordination/`:

| Canal | Uso |
|---|---|
| `PROTOCOL.md` | Reglas del protocolo (fuente de verdad) |
| `INBOX-CLAUDE.md` / `INBOX-CODEX.md` | Mensajería append-only entre agentes |
| `LEASES.md` | **Reclama el lease antes de editar.** Sin lease no se toca implementación ni tests |
| `CLAUDE-STATUS.md` / `CODEX-STATUS.md` | Heartbeat — single-writer por identidad |
| `CONTRACTS.md` | Cambios de contrato propuestos y sus ACK |
| `ASSIGNMENTS.md` / `PROGRESS.md` | Reparto de backlog y avance co-firmado |

Reglas mínimas: **no edites paths bajo lease ajeno**; los inbox y `CONTRACTS`/`KNOWLEDGE` son
append-only multiwriter (no admiten lease exclusivo); un rechazo se sustenta con evidencia
(hash del commit + tests), no con opinión.

Detalle: `.claude/specs/platform/codex-review-integration.md` (CTR-CODEX-REVIEW-001).

---

## 6. Perfiles de sandbox (Codex)

| Situación | Invocación |
|---|---|
| Auditoría read-only, sin red, secretos bloqueados | `codex exec -p audit -C "<repo>" "<prompt>"` |
| Trabajo con escritura acotada al workspace | `codex exec -p dev -C "<repo>" "<prompt>"` |

Toda clave de config **debe** validarse antes de usarse — rechaza campos inventados sin
llamar al modelo:

```powershell
codex exec --strict-config -c 'clave=valor' "ping"
codex debug models        # catálogo real; no confíes en nombres de modelo de guías
```

---

## 7. Definición de terminado

Una tarea no está terminada hasta que reportas:

1. **Archivos** creados / modificados / renombrados / eliminados.
2. **Tests que corriste y su salida real.** Si algo falla, se dice que falla y se pega la
   salida. Un test que no corriste no cuenta como verde.
3. **Gates afectados** — si tocaste `.claude/**`, `docs/**`, contratos, DAGs o migraciones,
   di qué gate corriste.
4. **Lo que quedó fuera** y por qué.
5. **Enlaces que pudieron romperse.**

No marques DONE por haber escrito el código. Se marca DONE cuando está verificado.

---

## 8. Honestidad

- No inventes fuentes, fechas, resultados, números ni referencias.
- Lo no verificado se etiqueta explícitamente: **Hipótesis** / **Inferencia** / **Pendiente
  de verificar**.
- Un backtest impresionante es sospechoso hasta demostrar lo contrario (§ constitución: Sharpe
  > 4-5 o DD < 1% ⇒ look-ahead o costos ignorados).
- Si una spec y el código se contradicen, **el código es el hecho** — repórtalo, no lo maquilles.
