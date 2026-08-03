# Canal de coordinación CLAUDE ↔ CODEX

> **Mapa del protocolo dual.** Este directorio no es documentación: es el **estado runtime**
> de dos agentes LLM trabajando en paralelo sobre la misma copia del repo. Se coordina
> exclusivamente por ficheros — no hay memoria compartida ni cola de mensajes.
>
> Por eso el gate de front matter lo trata como directorio **operacional** y lo exime del
> esquema tipado: exigir `last_verified` a un heartbeat que se reescribe cada 5 minutos
> obligaría a inventar metadatos.

---

## Los seis canales

| Fichero | Qué es | Escritura |
|---|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | **El prompt.** Se pasa idéntico a ambas LLM sustituyendo `{AGENT}`. Fuente de verdad del ciclo | Solo el operador |
| [`ASSIGNMENTS.md`](ASSIGNMENTS.md) | Reparto del backlog. CLAUDE = frontend/gobernanza/COP-producción · CODEX = DB/CI/identidad/métricas | Solo el operador |
| [`INBOX-CLAUDE.md`](INBOX-CLAUDE.md) · [`INBOX-CODEX.md`](INBOX-CODEX.md) | Mensajería dirigida. **Append-only, multiwriter** — no admiten lease exclusivo | Ambos |
| [`CLAUDE-STATUS.md`](CLAUDE-STATUS.md) · [`CODEX-STATUS.md`](CODEX-STATUS.md) | Heartbeat: estado, BLs activos, ficheros bloqueados, qué necesita del otro, qué deja en review. **Se actualiza cada ≤5 min** | Single-writer por identidad |
| [`LEASES.md`](LEASES.md) | **Reclamo de exclusividad sobre rutas.** Sin lease no se toca implementación ni tests | Ambos, append |
| [`CONTRACTS.md`](CONTRACTS.md) | Cambios de contrato propuestos y sus ACK. Append-only multiwriter | Ambos |
| [`PROGRESS.md`](PROGRESS.md) | Avance **co-firmado** por ambos agentes | Ambos |

Complementarios: [`KNOWLEDGE.md`](KNOWLEDGE.md) (reglas K-nnn acumuladas),
[`BASELINE.md`](BASELINE.md) (línea base de monitores),
[`INTEGRATION-AUDIT.md`](INTEGRATION-AUDIT.md),
[`PROTOCOL-COMMS-v2.md`](PROTOCOL-COMMS-v2.md) (staging de enmiendas, no vigente por sí solo).

Subdirectorios: `briefs/` (contexto por BL) · `reviews/` (review-packs) ·
`integration/` · `database/` · `monitor/` (logs, gitignorado) · `tmp/` (worktrees
efímeros, **gitignorado**).

---

## Las cuatro reglas que evitan colisiones

1. **Lease antes de editar.** Se anuncia la ruta en `LEASES.md` con identidad y expiración.
   No se tocan rutas bajo lease ajeno vigente. Los leases caducan: uno expirado no protege.
2. **Nunca implementes un BL del otro.** Si crees que hace falta, se propone en tu STATUS y
   se espera. Un rechazo vuelve a su dueño con razones — no lo arreglas tú.
3. **Nada es DONE hasta que el otro lo verifica.** El autor lo mueve a `para_review`; el
   veredicto lo emite la contraparte re-corriendo la sección "Verificación" del BL.
4. **Un rechazo se sustenta con evidencia** — hash del commit y salida de tests, no opinión.

## Higiene aprendida a golpes

- **`tmp/` está gitignorado por un incidente real**: un `git add .claude/coordination/` se
  llevó 681 ficheros de directorios temporales a un commit que debía tener tres. Hoy pesa
  varios GB en worktrees desechables. Nunca se versiona, y ni el gate de enlaces ni el de
  front matter lo escanean.
- **Este directorio está excluido del índice de Obsidian** salvo los ficheros de protocolo:
  indexar los worktrees metería miles de notas basura en el grafo.
- Los heartbeats reflejan el momento en que se escribieron. **Un STATUS de ayer con leases
  vencidos significa que el otro agente no está activo** — compruébalo antes de asumir
  que una ruta sigue reservada.

---

Contexto de por qué existe este canal y cómo se opera Codex:
[`../specs/platform/codex-review-integration.md`](../specs/platform/codex-review-integration.md).
Reglamento que ambos agentes leen al arrancar: [`../../AGENTS.md`](../../AGENTS.md).
