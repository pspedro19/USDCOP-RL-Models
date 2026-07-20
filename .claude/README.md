---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# `.claude/` — Project Knowledge Base

> Single entry-point to the USDCOP/Gold trading system's specs, rules, skills and agents.
> Start here. Every document carries typed YAML front matter (see **Conventions**), and the
> architectural counts below are **generated from source** — never hand-maintained.

---

## The one rule that governs everything: **auto-load boundary**

Claude Code injects **`rules/*.md` into every session** (always-on context budget). Everything
else in `.claude/` is **on-demand** (read only when a task needs it). Design accordingly:

| Folder | Loaded | Put here |
|--------|--------|----------|
| **`rules/`** | **Every session (auto)** | Thin, imperative, always-true rules & contracts. **Presupuesto: ≤3.000 palabras, verificado en CI.** |
| `specs/` | On demand | Dense reference (how things are built/wired). |
| `skills/` | On demand (invocadas) | Flujos operativos repetidos, ejecutables. |
| `agents/` | On demand (invocados) | Revisores especializados, **read-only**. |
| `experiments/` | On demand | RL experiment logs, queue, plans (process artifacts). |
| `templates/` | On demand | Scaffolds to copy when adding an asset / spec / experiment. |
| `generated/` | Nunca a mano | Inventario derivado del código (`inventory.json`). |

> Adding a big reference doc? It goes in `specs/`, **not** `rules/` — or it bloats every session.
> Only genuine always-apply rules (governance, contracts, gates, DO-NOTs) belong in `rules/`.

---

## Map — árbol real de `specs/` (generado)

> Escrito por `scripts/diagnostics/generate_inventory.py --write`. **No editar a mano.**
> Aquí vivía un árbol mantenido a mano que acabó describiendo un `assets/` con solo `xauusd/`
> mientras existían `btcusdt/` y `usdcop/` — y era el punto de entrada del que depende todo el
> esquema. Se eliminó a propósito: un índice manual del propio árbol siempre se queda atrás.

<!-- inv:specs_tree -->
```
. (2)
archive/2026-07/ (10)
assets/ (5)
assets/btcusdt/ (2)
assets/btcusdt/adr/ (1)
assets/btcusdt/design/ (3)
assets/btcusdt/design/adr/ (6)
assets/btcusdt/design/specs/ (13)
assets/btcusdt/specs/ (1)
assets/usdcop/ (3)
assets/xauusd/ (2)
assets/xauusd/adr/ (1)
assets/xauusd/specs/ (13)
audit/ (3)
data/ (1)
operations/ (2)
pipelines/ (3)
platform/ (20)
tracks/ (1)
tracks/news-analysis/ (14)
```
<!-- /inv -->

<!-- inv:knowledge -->
**9 rules** (~3,133 palabras auto-cargadas) · **106 specs** · **24 skills** · **3 agents**
<!-- /inv -->

---

## How to extend (scalability)

| I want to… | Do this |
|------------|---------|
| **Add a tradeable asset** (BTC, …) | Read `specs/assets/_onboarding-playbook.md`; copy `templates/asset-profile.example.yaml` → `config/assets/<id>.yaml`; create `specs/assets/<id>/` (or a single `<id>.md`) from `templates/spec-template.md`. |
| **Add a strategy track** | Create `specs/tracks/<track>.md` (or a folder) from `templates/spec-template.md`; register it in `rules/strategy-contract.md` (StrategyRegistry) and the dynamic registry. |
| **Add a reference spec** | `specs/<domain>/<name>.md` from `templates/spec-template.md`. Never add dense reference to `rules/`. |
| **Add an always-apply rule** | Append to the right `rules/*.md` (keep it thin). Update `rules/00-INDEX.md`. |
| **Run an experiment** | Copy `templates/experiment-config-template.md`; follow `rules/experiment-protocol.md`; log in `experiments/`. |

---

## Conventions

- **Naming**: kebab-case topic names. `NN-` prefix only inside ordered packages (`assets/xauusd/specs/SPEC-NN`, `tracks/news-analysis/NN_`).
- **Front matter obligatorio** (validado en CI por `test_knowledge_frontmatter.py`):

  ```yaml
  ---
  kind: rule | as-built | roadmap | adr | audit | historical
  status: IMPLEMENTED | PARTIAL | PLANNED | PAUSED | DEPRECATED | SUPERSEDED | HISTORICAL | ARCHIVED
  version: 1.0.0
  last_verified: YYYY-MM-DD     # "lo verifiqué contra el código", no "toqué el archivo"
  supersedes: []
  code_anchors: []              # rutas reales que la spec describe; si no existen, el gate falla
  ---
  ```

  `skills/` y `agents/` usan su propio front matter (`name` + `description`), que parsea el harness.
- **Ningún número vive en prosa.** Los conteos van en bloques `<!-- inv:key --> … <!-- /inv -->`
  que rellena `scripts/diagnostics/generate_inventory.py --write`.
- **SSOT ownership** (avoid duplication): freshness thresholds/recovery → `rules/data-freshness.md`; DAG schedule/timeline → `specs/operations/elite-operations.md`; strategy schemas → `rules/strategy-contract.md`; approval gates → `rules/approval-gates.md`. Other docs **link**, never re-tabulate.
- **`CLAUDE.md`** (repo root) is the always-loaded master; its "SDD Architecture" section indexes this tree.
