---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .gitignore
---

# BL-08 — Incidente .env en historial público: rotar+purgar+privatizar

**Fuente**: plan 03 §3.3 / memoria env-leak · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
.env real presente en commit ee91273 de un repo PÚBLICO (github.com/pspedro19/USDCOP-RL-Models). .env está ignorado hoy, pero el historial expone credenciales. BLOQUEA cualquier push.

## Qué falta exactamente
1) Rotar TODAS las keys expuestas (FRED, TwelveData×8, exchanges, Azure...). 2) `git filter-repo` purgando .env del historial. 3) Repo a privado (o decisión explícita). 4) Recién entonces habilitar push.

## Impacto frontend
Ninguno.

## Dependencias
— (bloquea el push de TODO lo demás).

## Verificación
`git log --all --full-history -- .env` vacío; keys viejas revocadas comprobado contra los proveedores.

## Notas constitución
Primera fila de la Readiness Matrix (BL-33). Evidencia, no intención.
