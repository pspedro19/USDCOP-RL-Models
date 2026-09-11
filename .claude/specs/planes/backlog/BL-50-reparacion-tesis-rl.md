---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-11
supersedes: []
code_anchors:
  - scripts/diagnostics/audit_thesis_rl_integrity.py
  - scripts/analysis/thesis_ppo_sanity.py
  - config/research/macro_availability.yaml
  - config/research/cost_contract.yaml
  - config/experiments/thesis_ppo_v2.yaml
  - src/research/features.py
  - src/research/session_gym.py
---

# BL-50 — Reparación y re-evaluación de la tesis RL (EXP-TESIS-RL-01)

**Fuente**: auditorías cruzadas del 2026-09-10 ·
**Ola**: — · **Esfuerzo**: L · **Trials**: 0 en reparación · +1 FT y +2 AT al reentrenar

## Contexto

Dos auditorías independientes —la de Codex
([informe](../../../../docs/analysis/exp-tesis-rl-01-auditoria-2026-09-10.md) con evidencia
hasheada y diagnóstico reproducible) y la de Claude— coincidieron en que **el rechazo
económico de la tesis se sostiene** pero **la implementación tenía defectos que impiden
concluir nada sobre el potencial**, y que el documento de resultados contenía errores de
convención estadística. El corrigendum vive en
[`06-RESULTADOS.md`](../06-RESULTADOS.md); el programa de re-evaluación en
[`06-PRE-REGISTRATION-v3.md`](../06-PRE-REGISTRATION-v3.md).

Esta ficha es el paraguas de ejecución. Absorbe [`BL-49`](BL-49-tests-2-y-14-tesis.md) (sus
dos tests entran aquí) y re-enfoca [`BL-48`](BL-48-costos-ejecucion-intradia.md) (la línea de
ejecución pasiva queda suspendida hasta tener un venue con libro de órdenes).

## Etapas y estado

| Etapa | Contenido | Estado |
|---|---|---|
| 1. Fuentes y costos | `macro_availability.yaml` (regla de disponibilidad), `cost_contract.yaml` con unidad declarada, máscara v2 con reglas §6.3 y festivos de EE. UU., schema v2 (39 → 37 features) | HECHO |
| 2. Entorno | Fuga macro cerrada, costo terminal en el reward y valorado en la barra 59, ventanas intra-sesión, identidad del dataset por sha256, tests HMM no tautológicos, specs parciales para el carril live | HECHO |
| 3. Sanidad del optimizador | `thesis_ppo_sanity.py` con fixtures S1-S4 de solución conocida y sondas ordenadas | EN CURSO |
| 4. Reentreno sin fugas y juez forward | v3 firmado, entreno v2, baselines que faltaban, carril forward sellando de verdad | PENDIENTE |
| 5. Corrigendum | Correcciones de §0-§7 con cifras de la evidencia | HECHO |
| 6. Gobernanza | Contabilidad de trials, coordinación, cierre de BL-48/49 | PENDIENTE |

## Criterio de cierre

1. `scripts/diagnostics/audit_thesis_rl_integrity.py` reporta `causality_gate = True`
   (cumplido el 2026-09-11).
2. Los fixtures S1-S4 pasan con **una** receta congelada, declarada antes de tocar datos de
   mercado.
3. El pre-registro v3 está firmado **antes** de la primera corrida v2, y la contabilidad de
   trials queda conciliada en el registro del activo.
4. El carril forward sella decisiones con información realmente disponible, verificable por
   cadena de hashes.

## Lo que esta ficha no autoriza

No promueve ninguna estrategia, no reabre el hold-out como juez confirmatorio de la versión
corregida y no convierte los pases sintéticos de la Etapa 3 en evidencia sobre el mercado.
