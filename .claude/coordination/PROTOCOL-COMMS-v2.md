# PROTOCOL-COMMS v2 — Cómo setear la comunicación dual CLAUDE↔CODEX (destilado y verificable)

> Documento-prompt REUTILIZABLE para bootstrapear cualquier sesión dual futura.
> Destila las 22 lecciones K-001..K-022 ganadas por hechos el 2026-07-27.
> Estado: DRAFT-CLAUDE — CODEX: verifica sección por sección y añade tu firma/objeciones al pie.

## 1. Setup inicial (orden exacto, ~2 min)

1. **Una raíz por identidad** con `instance_id` único en su STATUS (K-004-codex: dos
   raíces = overwrite real). Subagentes: nunca escriben STATUS ni commitean.
2. Cada raíz monta EN SU PRIMER CICLO sus 3 monitores:
   - **Mensajes** (≤30s): hash-watch de {STATUS-del-otro, INBOX-propio, CONTRACTS,
     KNOWLEDGE} → evento inmediato, no polling manual (K-007).
   - **Logs** de SU dominio (K-019): Claude=dashboard+scheduler; Codex=postgres+DAGs.
     Filtros selectivos de error, nunca raw.
   - **Dispatcher** (~2min): flota <mínimo && BLs PLANNED → "lanzar tanda"; actividad
     del otro por **mtime de canales, no campos internos** (aprendido: el campo
     timestamp: se olvida bajo carga).
3. Verificar el par ANTES de trabajar: escribir línea de LOG propia y confirmar que el
   watcher del otro la reporta (<60s). Sin eco confirmado = no arrancar tandas.

## 2. Canales y su semántica (nunca mezclar)

| Canal | Semántica | Regla |
|---|---|---|
| `INBOX-{OTRO}.md` | urgente/accionable, 1 línea/ítem | se lee ANTES que todo; quien procesa tacha |
| `{YO}-STATUS.md` | heartbeat ≤5min + LOG append | solo su dueño escribe |
| `CONTRACTS.md` | C-NNN PROPOSED→ACK/OBJECION→APPLIED/REMEDIADO | append-only; espejo Py↔TS mismo commit; auto-ACK 15min SOLO aditivo |
| `KNOWLEDGE.md` | K-NNN aprendizajes/propuestas por HECHOS | ambos escriben; colisión de número = orden de archivo gana, se anota sin reescribir (K-009) |
| `LEASES.md` | ruta+dueño+instance_id+expira ≤45min | segunda instancia que ve lease vigente → STOP |
| `briefs/BL-XX.md` | contexto pre-masticado con FUENTE verificable (path:line/query+timestamp) | conteos sin fuente envejecen (K-005: 3/16 vs 2/15, dos veces probado) |
| `reviews/BL-XX.md` | pack INMUTABLE: hash+paths+C-NNN+comandos+resultados+delta-vs-BASELINE | review contra el HASH, jamás working-tree móvil (K-002/K-006) |
| `PROGRESS.md` | tablero conjunto co-firmado | métrica = BLs APROBADOS/tiempo, no commits (K-015) |
| `BASELINE.md` | fallos pre-existentes conocidos por monitor | aprobar por DELTA, no por total |

## 3. El ciclo con adversarialidad integrada

```
leer INBOX → responder ACKs → leer STATUS del otro → tanda (N agentes, archivos
disjuntos, leases declarados) → SELF-RED-TEAM (1 refutador propio por tanda, K-013)
→ commit [yo] BL-XX Ref C-NNN|C-EXEMPT → review-pack → para_review → cross-review
del otro (APROBADO/RECHAZADO con razones) → concesión ⇒ SIEMPRE regla K derivada
(K-022: el golpe se codifica para matar la clase entera de error)
```

## 4. Reglas duras anti-colisión (todas pagadas con incidentes reales)

- git `index.lock`: retry backoff 10s×6; commits frecuentes y pequeños (K-011).
- PROHIBIDO `git stash` en la raíz compartida (K-018: pop abortado por locks).
- Ownership: el prompt de cada tanda CITA ASSIGNMENTS; BL ajeno detectado se CEDE con
  el trabajo como borrador del dueño, no se defiende (K-021).
- Scope: "entregado" se mide contra el TEXTO del BL; splits de fase se declaran ANTES
  en el review-pack (K-012).
- Contrato compartido sin C-NNN en el commit = rechazo automático en review.

## 5. Escalamiento

- Sin eco del otro >15min (mtime real): seguir solo con BLs independientes + nota.
- >60min: estado BLOCKED + pregunta CONCRETA al operador en STATUS. Jamás inventar.
- Decisión de MODELADO detectada: PARA ambos — el operador pre-registra (0 trials).

## 6. Verificación de este documento (checklist para CODEX)

- [ ] §1: ¿tu setup real coincide? (instance_id, 3 monitores, eco confirmado)
- [ ] §2: ¿algún canal con semántica distinta a como lo usas?
- [ ] §3: ¿tu self-red-team corre ANTES de tu para_review?
- [ ] §4: ¿incidentes tuyos no capturados que falten como regla?
- [ ] §5: ¿umbrales 15/60min correctos para tu cadencia?
- FIRMA CODEX + fecha, u OBJECIONES numeradas debajo:

---
FIRMA CLAUDE: claude-root-da4532c6 · 2026-07-27
FIRMA CODEX: (pendiente verificación)
