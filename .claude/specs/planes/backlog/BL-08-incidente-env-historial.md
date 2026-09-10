---
kind: roadmap
status: PARTIAL
version: 1.3.0
last_verified: 2026-09-10
supersedes: []
code_anchors:
  - .gitignore
  - config/governance/security_incident_env_history.yaml
  - tests/regression/test_bl08_env_history_control.py
---

# BL-08 — Incidente .env en historial público: rotar+purgar+privatizar

**Fuente**: plan 03 §3.3 / memoria env-leak · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
.env real presente en commit ee91273 de un repo PÚBLICO (github.com/pspedro19/USDCOP-RL-Models). .env está ignorado hoy, pero el historial expone credenciales. BLOQUEA cualquier push.

## Entrega parcial verificada 2026-08-05

Existe un control trackeado en `config/governance/security_incident_env_history.yaml` que
registra el incidente sin copiar valores secretos y mantiene `push_allowed: false`. La evidencia
local medida declara cuatro hechos separados: `.env` no está trackeado en el árbol actual, sí
aparece en el historial local, su blob histórico sigue siendo alcanzable y este clon no ha sido
reescrito. `tests/regression/test_bl08_env_history_control.py` compara esos campos con metadatos
Git en ambas direcciones, sin materializar el contenido, y exige un checkout con historial completo.

La visibilidad remota no se deriva de ese gate local. El valor `public` permanece identificado
como `OPERATOR_ATTESTATION` hasta que el operador ejecute y atestigüe el cambio de visibilidad.

Esto no cierra el incidente: la rotación y revocación contra cada proveedor, la purga del remoto
y la decisión de visibilidad siguen marcadas como acciones del operador sin evidencia. Por eso el
estado correcto es `PARTIAL`, nunca `IMPLEMENTED`, y la prohibición de push permanece intacta.

## Decisión del operador — 2026-08-24: aplazar la rotación

Preguntado explícitamente, el operador responde que está en **entorno de pruebas** y no quiere
rotar las claves por ahora; pide recordatorio para más adelante. Queda registrado en
`config/governance/security_incident_env_history.yaml::operator_decision_log` y en la memoria
de proyecto.

Lo que el aplazamiento **no** cambia: las credenciales están en el historial **público** de
GitHub, no en el entorno local — el entorno acota dónde se usan, no quién puede leerlas. Por
eso las cuatro `operator_actions` siguen en `complete: false` y `push_allowed` sigue en `false`.

## Por qué `PARTIAL` y no `IMPLEMENTED` (2026-08-24)

**Ninguno de esos 4 pasos está hecho.** El estado subió de `PLANNED` a `PARTIAL` porque el
deliverable que ASSIGNMENTS asigna a este BL —el *checklist .env con el operador*— sí
existe y está trackeado: `config/governance/security_incident_env_history.yaml`, con
`operator_actions` (las 4 acciones, todas `complete: false`), las revisiones públicas
afectadas (`ee91273`, `1d41812`) y `release_policy.push_allowed: false`.

`PARTIAL` significa aquí **"el registro del incidente shippeó"**, no "la remediación
empezó". El fichero de gobierno declara `status: BLOCKED_OPERATOR` y es la fuente de
verdad del avance real; este documento solo refleja que su artefacto ya está en el repo.

**El incidente sigue abierto y `git push` sigue prohibido** (`ASSIGNMENTS.md:29`). Rotar
credenciales, purgar el historial y decidir la visibilidad del repositorio son acciones del
operador; hasta que `operator_actions` tenga sus cuatro `complete: true` con evidencia,
este BL no pasa a `IMPLEMENTED`.

## Qué falta exactamente
1) Rotar TODAS las keys expuestas (FRED, TwelveData×8, exchanges, Azure...). 2) `git filter-repo` purgando .env del historial. 3) Repo a privado (o decisión explícita). 4) Recién entonces habilitar push.

## Impacto frontend
Ninguno.

## Dependencias
— (bloquea el push de TODO lo demás).

## Verificación
El gate local debe pasar contra un clon completo. Para cerrar el incidente: `git log --all
--full-history -- .env` vacío y objetos históricos no alcanzables; keys viejas revocadas
comprobado contra los proveedores; visibilidad remota atestiguada por el operador. Solo entonces
puede revisarse `push_allowed: false`.

## Notas constitución
Primera fila de la Readiness Matrix (BL-33). Evidencia, no intención.
