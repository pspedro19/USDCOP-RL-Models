---
kind: spec
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - .claude/coordination/PROTOCOL.md
  - .claude/coordination/PROGRESS.md
---

# Canal de INTEGRACIÓN — auditoría cruzada CLAUDE ↔ CODEX

> Creado por `claude-root-9c3f1e42` el 2026-07-28 por orden del operador.
> **Propuesto a CODEX; queda ACTIVE cuando él cofirme al final de este archivo.**
> Es el único lugar donde viven los resúmenes de integración. No duplica los
> canales existentes: `INBOX-*` sigue siendo para mensajes, `PROGRESS.md` para el
> tablero, `reviews/BL-XX.md` para los packs por BL contra hash.

---

## 1. Por qué existe

Terminada la FASE I (implementación de ambos lotes), el riesgo ya no es "falta
código": es que **las dos mitades no encajen**, que cada uno haya resuelto el
mismo problema dos veces, y que los tests verdes de cada lado no signifiquen que
el sistema funciona junto. Este canal existe para tres cosas y solo tres:

1. **Auditarnos mutuamente** de forma adversarial (el default es rechazar).
2. **Encontrar las fronteras** donde los dos lotes se tocan y fijar qué debe ser
   verdad en cada una.
3. **Acordar la batería BDD/TDD** que cierra la FASE II.

## 2. Ficheros del canal

| Fichero | Dueño de escritura | Qué contiene |
|---|---|---|
| `README.md` (este) | ambos, append-only al final | protocolo del canal + cofirmas |
| `AUDIT-CLAUDE-of-CODEX-*.md` | CLAUDE | mis hallazgos sobre el trabajo de CODEX |
| `AUDIT-CODEX-of-CLAUDE-*.md` | CODEX | los suyos sobre el mío |
| `SELF-REDTEAM-CLAUDE.md` | CLAUDE | lo que encuentro contra MI PROPIO trabajo |
| `SELF-REDTEAM-CODEX.md` | CODEX | ídem por su lado |
| `BDD-MATRIX.md` | ambos, por secciones marcadas | escenarios Gherkin por BL, con la invariante que protege |
| `TDD-GAPS.md` | ambos | tests que faltan, con la mutación que debe ponerlos rojos |
| `INTEGRATION-CONTRACT.md` | ambos, por frontera | dónde se tocan los lotes y qué se exige en cada punto |

**Regla de escritura**: nadie edita el fichero de auditoría del otro. Se responde
con una sección `## RESPUESTA <owner>` al final del mismo fichero, append-only,
o con un fichero propio. La historia no se reescribe.

## 3. Formato de hallazgo (obligatorio)

Un hallazgo sin escenario de fallo concreto no es un hallazgo, es una opinión.

```
### <ID>  ·  <BLOQUEANTE|GRAVE|MENOR|OBSERVACIÓN>
**Dónde**: <fichero>:<línea>
**Qué está mal**: <una frase>
**Escenario de fallo**: <entradas/estado concretos → daño concreto>
**Evidencia**: <salida real ejecutada, o "ESTÁTICO" si es por lectura>
**Remedio propuesto**: <qué haría falta>
**Test que lo cerraría**: <nombre + mutación que debe ponerlo rojo>
```

Severidad, con criterio compartido:

- **BLOQUEANTE** — pierde dinero, expone datos, corrompe estado, o publica un
  claim falso. No se integra hasta arreglarlo.
- **GRAVE** — rompe una invariante de `.claude/rules/` sin daño inmediato.
- **MENOR** — deuda real (SOLID/DRY/clean code) que costará más si se deja.
- **OBSERVACIÓN** — mejora, sin defecto.

**Si no puedes construir el escenario de fallo, baja la severidad.** Y si algo
está bien hecho, dilo explícitamente: el objetivo es que el sistema funcione, no
ganar la discusión.

## 4. Estándar que se exige a ambos lados

| Principio | Qué se verifica en la auditoría |
|---|---|
| **SSOT** | un concepto, una fuente. Toda duplicación entre lotes es hallazgo. |
| **DRY** | dos implementaciones del mismo cálculo = el backtest miente en algún sitio. |
| **SOLID** | una responsabilidad por módulo; extensión sin tocar el core (motor nuevo ⇒ cero cambios en el evaluador). |
| **Fail-closed** | entrada ausente/stale/inválida ⇒ BLOQUEA. Un default silencioso en trading es GRAVE. |
| **Contract-first + espejo** | ningún contrato en un solo lenguaje; Py y TS rechazan EXACTAMENTE lo mismo. |
| **Tests que muerden** | todo test declara la mutación que lo pone rojo. Un test que pasa con el código roto es peor que ninguno. |
| **Honestidad** | ningún número sin fuente publicada; N<20 ⇒ solo conteo y PnL; jamás `Infinity`/`NaN` en JSON; ningún "verificado" que no se ejecutó. |
| **0 trials** | ninguna decisión de modelado se toma implementando ni testeando. Se declara al operador. |

## 5. Ciclo de la FASE II

```
1. AUDITAR    cada uno publica su auditoría del otro + su self-red-team.
2. TRIAGE     se clasifican los hallazgos por severidad; los BLOQUEANTES
              se reparten al DUEÑO del código (nadie arregla el del otro).
3. ROJO       se escribe el test que falla ANTES del fix. Se pega el rojo.
4. VERDE      se arregla. Se pega el verde. Mutación demostrada.
5. FRONTERAS  se cierra `INTEGRATION-CONTRACT.md`: cada punto de contacto con
              su test de integración verde.
6. CAMPAÑA    batería completa (pytest + Vitest + Playwright/BDD + monitores:
              frontmatter, manifests, scripts-layout, rbac:check) sobre un
              hash sellado, con DELTA vs `BASELINE.md`.
7. CIERRE     cross-review por hash → recién ahí un BL pasa a DONE.
```

Nada pasa a DONE por tener tests verdes en su propio lane: hace falta la
revisión del otro contra un hash inmutable. Esa regla no cambia.

## 6. Qué NO se difiere aunque vayamos rápido

- Constitución quant: 0 trials, DSR bar 0.95 sin ADR, ninguna decisión de
  modelado tomada por un ingeniero.
- `git push` PROHIBIDO hasta cerrar BL-08.
- BL-41: sin DDL ni cutover hasta Vault real, roles no-super y TDD.
- Fronteras de `ASSIGNMENTS.md` y leases antes de escribir.
- Allowlist explícita de rutas + `git diff --cached --name-status` justo antes de
  cada commit (regla permanente tras el incidente de índice `CXD-045`).

---

## FIRMAS

- `claude-root-9c3f1e42` · 2026-07-28 · **PROPUESTO**. Abro el canal con mis dos
  auditorías del trabajo de CODEX (migraciones y módulos), mi self-red-team y la
  matriz BDD/TDD inicial. Corrige u objeta lo que quieras de este protocolo: es
  un borrador para trabajar, no un decreto.
- `codex-root-880ff498` · _pendiente de cofirma_
