---
kind: as-built
status: PARTIAL
contract: CTR-CODEX-REVIEW-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# SDD Spec: Codex como revisor independiente (Claude piloto)

> **Responsibility**: Fuente autoritativa de cómo este repo usa **Codex CLI** como revisor
> independiente del código que escribe Claude Code — instalación, config validada contra el
> binario, aislamiento de secretos, y el loop de auditoría. Otras specs enlazan aquí; no
> dupliquen flags ni config.
>
> Contract: CTR-CODEX-REVIEW-001
> Version: 2.0.0
> Date: 2026-07-20
> Status: IMPLEMENTED (CLI + auth + perfil de auditoría verificados end-to-end)
> Cross-refs: `../../rules/quant-constitution.md`, `cicd-testing.md`, `../audit/`

---

## 1. Modelo mental

```
Claude Code  (PILOTO)      implementa, orquesta, decide. SDD + TDD.
      │
      ▼
Codex plugin (REVISOR)     loop interactivo dentro de Claude Code
      │                    /codex:review · /codex:adversarial-review · /codex:rescue
      ▼
Codex CLI    (MOTOR)       mismo binario, misma auth, misma config
                           headless: `codex review` · `codex exec`
```

El plugin **no** es un runtime aparte: delega en el binario local. La config de `~/.codex/`
gobierna por igual al plugin y a `codex exec`.

| Situación | Herramienta | Perfil | Reasoning |
|---|---|---|---|
| Implementar una feature | Claude Code | — | — |
| Revisión estándar | `/codex:review --background` | `-p audit` | high |
| Revisión crítica pre-merge | `/codex:adversarial-review` | `-p audit` | high |
| Auditoría profunda headless | `codex exec -p audit` | `-p audit` | high |
| Codex arregla un bug | `/codex:rescue` (worktree aislado) | perfil aparte con escritura | medium/high |

---

## 2. As-built verificado (2026-07-20)

| Pieza | Estado | Detalle |
|---|---|---|
| Codex CLI | ✅ | `0.144.6`, install npm, `codex doctor` = 17 ok · 0 fail |
| Auth | ✅ | `chatgpt` OAuth, `~/.codex/auth.json` |
| `~/.codex/audit.config.toml` | ✅ **creado y validado** | perfil `repo-audit`, ver §4 |
| Bloqueo de `.env` | ✅ **verificado empíricamente** | ver §5 |
| `~/.codex/config.toml` (base) | ⚠️ **no creado — a propósito** | ver §6 (footgun) |
| `.codex/config.toml` (proyecto) | ⚠️ no creado — a propósito | mayor precedencia que el perfil; se omite durante la 1ª auditoría |
| Plugin `codex@openai-codex` | ⚠️ no instalado | repo `openai/codex-plugin-cc` confirmado |

---

## 3. Método de validación (obligatorio)

Toda clave de config **debe** pasar por `--strict-config`, que rechaza campos desconocidos
**antes de llamar al modelo** — validación gratis:

```powershell
codex exec --strict-config -c 'clave=valor' "ping"
```

El catálogo de modelos se resuelve local y gratis con:

```powershell
codex debug models
```

> Estas dos herramientas existen precisamente porque las guías que circulan traen campos y
> modelos inventados. **Ninguna clave entra a esta spec sin pasar por ahí.**

### 3.1 Modelos reales en este binario

`codex debug models` → **`gpt-5.6-sol`**, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`,
`codex-auto-review`.

- `gpt-5.6-sol`: default reasoning **`low`**; soporta `low|medium|high|xhigh|max|ultra`.
- `gpt-5.4-mini`: **existe** (una revisión previa de esta spec lo declaró inexistente — era
  incorrecto; el catálogo manda).
- `gpt-5.6-terra` / `gpt-5.6-luna`: **NO existen** en este catálogo. No usarlos.

### 3.2 Correcciones a config que circula en guías internas

| Escrito en guías | Veredicto | Correcto |
|---|---|---|
| `[deny_read_paths] patterns=[…]` | ❌ `unknown configuration field` | Sistema `[permissions.*]` (§4) |
| `sandbox = "elevated"` (nivel raíz) | ❌ inválido en raíz | `[windows] sandbox = "elevated"` ✅ clave válida (§5.2) |
| `codex exec --ask-for-approval never` | ❌ el flag **no existe en `exec`** | `approval_policy` en config / perfil |
| `codex sandbox windows …` | ❌ no existe ese subcomando | `codex sandbox -p <perfil> -P <permission-profile> -- <cmd>` |
| `gpt-5.4-mini` no vigente | ❌ **sí existe** | ver §3.1 |
| `gpt-5.6-terra` / `luna` | ❌ no existen | usar `gpt-5.6-sol` |
| `[profiles.x]` en `config.toml` | ❌ desactualizado | archivos `~/.codex/<name>.config.toml` + `-p` |

---

## 4. El perfil de auditoría — `~/.codex/audit.config.toml`

Read-only, sin red, con lectura de secretos denegada. **Es el único archivo de config creado**
(sin base ni config de proyecto, para eliminar la variable de precedencia en la 1ª auditoría).

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "high"
approval_policy = "never"

default_permissions = "repo-audit"

[windows]
sandbox = "elevated"

[permissions.repo-audit]
description = "Read-only repository audit with secret files blocked"
extends = ":read-only"

[permissions.repo-audit.filesystem]
":root" = "deny"
":minimal" = "read"
glob_scan_max_depth = 6

[permissions.repo-audit.filesystem.":workspace_roots"]
"." = "read"
".env" = "deny"
".env.*" = "deny"
"*/.env" = "deny"
# … profundidades explícitas + *.pem, *.key, credentials*.json, secrets/ …

[permissions.repo-audit.network]
enabled = false
```

Notas de implementación verificadas:

- **No mezclar** `[permissions.*]` con `sandbox_mode`/`sandbox_workspace_write`. El perfil no
  fija `sandbox_mode`; `extends = ":read-only"` ya produce `sandbox: read-only` en la salida.
- **Nada de `**` en los globs.** Codex advierte: *"Non-macOS sandboxing does not support
  unbounded `**` natively"*. Se usan **profundidades explícitas** (`*/`, `*/*/`) más
  `glob_scan_max_depth`. Con `**` los warnings inundan cada corrida.

Uso:

```powershell
codex exec -p audit -C "<repo>" "<prompt de auditoría>"
```

---

## 5. Aislamiento de secretos — RESUELTO y verificado

Este repo tiene `.env` real en la raíz y antecedente de fuga pública (memoria
`env-leak-github-public`), así que el bloqueo no se asume: se prueba.

### 5.1 Evidencia

Protocolo: **nunca testear con el secreto real** — si el bloqueo falla, el propio test lo
filtra. Se usan archivos señuelo con el mismo patrón, y luego se borran.

| Test | Resultado | Veredicto |
|---|---|---|
| `README.md` → primer heading | `# USDCOP Trading System` | ✅ lectura normal funciona |
| `CLAUDE.md` → nº de líneas | `381` | ✅ el repo es auditable |
| `.env.sandboxtest` (señuelo, patrón `.env.*`) | `BLOCKED` | ✅ denegado |
| `.env` exacto (señuelo en dir scratch) | `BLOCKED` | ✅ denegado |

El control positivo es tan importante como el negativo: sin él, un perfil que bloquea *todo*
pasaría por "seguro" y produciría una auditoría vacía.

### 5.2 El matiz del backend elevado de Windows

`codex sandbox` (ejecutar comandos shell arbitrarios bajo sandbox restringido) falla con:

> `windows sandbox failed: Restricted read-only access requires the elevated Windows sandbox backend`

`[windows] sandbox = "elevated"` es clave válida, pero el backend **requiere privilegios de
administrador** — esta sesión corre como no-admin y el intento se cuelga esperando UAC.

**Esto NO afecta la protección de secretos**: las reglas `deny-read` **sí se aplican a las
lecturas del propio agente** en `codex exec -p audit` (probado arriba). La limitación aplica
solo al subcomando `codex sandbox`. Si en el futuro se necesita `codex sandbox`, habrá que
correr una consola elevada.

---

## 6. Footgun conocido: `codex` sin `-p audit`

**No existe `~/.codex/config.toml` base**, por diseño. Consecuencia: una invocación de
`codex` / `codex exec` **sin `-p audit`** corre con los defaults del binario —
`reasoning: none` y **sin las reglas deny** — y en ese modo puede leer `.env`.

| Invocación | Protegido | Reasoning |
|---|---|---|
| `codex exec -p audit …` | ✅ sí | high |
| `codex exec …` (sin perfil) | ❌ **no** | none |

Opciones para cerrarlo (decisión del operador, no tomada aún):

1. Replicar el bloque `[permissions.*]` en `~/.codex/config.toml` para que el default también
   sea seguro. Cuesta: reintroduce una capa de precedencia.
2. Mantener la disciplina de **siempre** pasar `-p audit`.

---

## 7. Ejecutar la auditoría

### 7.1 `codex review` es orientado a diff — no sirve para auditoría profunda

```powershell
codex review --uncommitted        # staged + unstaged + untracked
codex review --base main          # rama actual contra base
codex review --commit <SHA>
```

⚠️ **Estado actual del repo: rama `main`, con 3 archivos modificados.** Por lo tanto
`codex review --base main` compara `main` contra sí misma ⇒ **diff vacío**. Para auditar el
sistema completo hay que usar `codex exec -p audit` con un prompt de auditoría, o revisar
una rama de trabajo contra `main`.

### 7.2 Auditoría profunda (la vía correcta acá)

```powershell
codex exec -p audit -C "<repo>" `
  "Audita <área> contra .claude/specs/<spec>.md y .claude/rules/. No modifiques archivos.
   Reporta por severidad, citando archivo:línea. Marca como hipótesis lo no verificado."
```

Flags útiles: `--json`, `-o <FILE>` (guardar salida como evidencia), `--output-schema`.

### 7.3 Gotcha de exit code

Con stdin cerrado (ejecución desde herramienta), `codex exec` imprime
`Reading additional input from stdin...` y sale con **255** aunque la corrida haya sido
correcta. **Juzgar por la salida, no por el exit code.**

---

## 8. Loop de trabajo

1. **Claude construye** (SDD+TDD): comportamiento actual → criterios de aceptación → tests en
   Red → cambio mínimo en Green → tests. **Sin commit.**
2. **Codex revisa en paralelo**: `/codex:review --background` → `/codex:status` →
   `/codex:result`. Read-only garantizado por sandbox, no por prompt.
3. **Pre-merge crítico** (auth, migraciones, RBAC, órdenes/ejecución, aislamiento de tenant,
   registry, Docker): `/codex:adversarial-review --base main --background`.
4. **Rescate** tras 2 intentos fallidos: `/codex:rescue` en **worktree aislado**, con escritura
   escalada solo en esa invocación.
5. **Transferencia**: `/codex:transfer` → `codex resume <session-id>`.

### Resolución de desacuerdos (vinculante)

Cuando Codex y Claude difieren, **no se decide por confianza del modelo**: se identifica la
suposición en disputa y se decide por **evidencia** — un test que falle o pase.

Los hallazgos de Codex son **hipótesis, no verdades**. Misma disciplina que
`quant-constitution.md` §2 aplica a los claims de edge: sin verificación contra el código o un
test, un hallazgo no entra al backlog como confirmado.

---

## 9. Review-gate automático — NO activar por defecto

```text
/codex:setup --enable-review-gate
/codex:setup --disable-review-gate
```

Hook `Stop` que bloquea el fin de turno si Codex halla issues. El propio plugin advierte que
puede generar ciclos largos Claude↔Codex y **agotar límites de uso rápido**. Solo en sesión
crítica supervisada; desactivar al terminar.

---

## 10. Instalación del plugin (pendiente)

```text
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

`/reload-plugins` no es opcional. Humo: `/codex:review --background` → `/codex:status` →
`/codex:result`.

---

## Cross-References

| Concern | Spec |
|---------|------|
| Hallazgos como hipótesis, anti-selección | `../../rules/quant-constitution.md` |
| CI/CD, gates de cobertura | `cicd-testing.md` |
| Auditorías point-in-time → backlog P0/P1/P2 | `../audit/AUDIT-2026-07-remediation.md` |
| RBAC / superficies a revisar adversarialmente | `../../rules/rbac.md` |

## DO NOT

- Do NOT documentar una clave de config sin `--strict-config`, ni un modelo sin `codex debug models`.
- Do NOT correr Codex sobre este repo **sin `-p audit`** — sin el perfil, `.env` es legible (§6).
- Do NOT testear el bloqueo de secretos con el secreto real — usar señuelos y borrarlos (§5.1).
- Do NOT validar un perfil solo con el test negativo — sin control positivo, un perfil que
  bloquea todo parece seguro y produce auditorías vacías.
- Do NOT usar `**` en globs de `permissions.filesystem` — usar profundidades explícitas.
- Do NOT mezclar `[permissions.*]` con `sandbox_mode` / `sandbox_workspace_write`.
- Do NOT usar `--ask-for-approval` en `codex exec` (no existe) ni `[profiles.x]` (desactualizado).
- Do NOT esperar que `codex review --base main` audite algo estando en `main` — diff vacío.
- Do NOT tratar un hallazgo de Codex como confirmado sin evidencia.
- Do NOT dejar el review-gate activo entre sesiones.
