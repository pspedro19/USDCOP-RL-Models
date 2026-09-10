---
kind: adr
status: IMPLEMENTED
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
last_verified: 2026-08-24
supersedes: []
code_anchors:
  - .claude/rules/quant-constitution.md
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - .claude/specs/planes/06-tesis-rl-llm-hibrido.md
  - config/research/partition.yaml
  - services/common/metrics.py
  - tests/regression/test_trial_ledger.py
---

# ADR-0023 — El carril de tesis hereda el contador de trials de USD/COP

## Contexto

El plan de tesis (`.claude/specs/planes/06-tesis-rl-llm-hibrido.md`) define en §11.12 su
propia contabilidad de `N_trials` para deflactar el Deflated Sharpe. Escrito de forma
aislada, ese conteo arrancaría en cero: la tesis es un trabajo nuevo, con partición nueva,
modelos nuevos y hold-out propio.

**Ese cero no existe.** El activo ya tiene historia:

- `.claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md:15` declara `n_trials_total: 111`
  para USD/COP (reconciliación BL-12-r2, 2026-07-27).
- La línea 1604 del mismo registro lo dice sin rodeos:
  *«`n_trials_total` sigue siendo la suma de ambos linajes y **la partición jamás resetea N**»*.
- La constitución (`quant-constitution.md` §2) obliga a que **cada versión, cada grid, cada
  gate mirado** cuente como trial, y `ADR-0022` fija que el conteo del activo es la suma de
  los linajes FT (predictivo) y AT (económico).

Además, el hold-out de la tesis (2024-01-01 → 2026-08-24, 677 sesiones, congelado en
`config/research/partition.yaml`) **incluye 2025**, que el track H5 de producción ya
grid-searcheó: el «42-cell grid, #8 of 42» que documenta `CLAUDE.md`. Ese bloque se
mantiene dentro del hold-out a conciencia, para ganar la potencia estadística que §11.2
exige (677 supera el umbral de 500 que el propio plan llama «el límite»); pero mantenerlo
sin declarar la deuda sería presentarlo como virgen cuando no lo es.

Si la tesis llevara su propio contador, tendríamos **dos verdades sobre el mismo activo**:
el registro constitucional diciendo 111+ y el capítulo de resultados diciendo N pequeño,
con un DSR artificialmente favorable. Es exactamente la clase de contabilidad separada que
la constitución existe para impedir.

## Decisión

**El carril de tesis no lleva contador propio. Usa el del activo.**

1. `N_trials` de cualquier claim de la tesis = **111 heredados + los trials de la tesis**
   (los 60 de Optuna en F5, las ablaciones factoriales, el grid híbrido de 16 celdas de
   §10.5, los baselines y controles de §10.6/§10.8).
2. Los trials nuevos se registran en el `HYPOTHESIS-REGISTRY` de USD/COP con su linaje
   FT/AT y su `research_cluster`, igual que cualquier otro (ADR-0022). No hay un registro
   paralelo «de tesis».
3. El DSR (`services/common/metrics.py::deflated_sharpe_ratio`) se recomputa **siempre**
   con el total, nunca con el subconjunto de la tesis.
4. La herencia se declara **ex-ante**, antes de mirar ningún resultado: está escrita en
   `config/research/partition.yaml` bajo `trials.inherited_n: 111` y en el pre-registro
   del hold-out.
5. La contaminación parcial del hold-out se declara en el mismo sitio
   (`trials.holdout_partially_looked_at: true`) con la razón por la que se acepta.

## Consecuencias

**Asumidas, no lamentadas.** Con N ≥ 111 el bar del DSR es duro, y es posible que ningún
brazo de la tesis lo supere. El plan ya lo previó por dos caminos:

- §11.8 define el DSR como **«cantidad reportada, no compuerta binaria»** — se publica el
  número con su regla de interpretación, no un aprobado/suspenso.
- §11.1 y §3.9 obligan a reportar un contraste indecidible **como indecidible**, no como
  «sin diferencias significativas».
- §3.6 declara que **los resultados negativos son válidos**, y la constitución §3 que si
  una estrategia no bate al baseline tonto, *el baseline es la estrategia*.

Una tesis que reporta honestamente «con el N real del activo, el efecto no es distinguible
de cero» es un resultado. Una que reporta un DSR inflado por haber reseteado el contador
es un error metodológico — precisamente el que la auditoría de julio de 2026 encontró en el
track de producción y que originó la constitución transversal.

**Lo que NO cambia**: la partición sigue congelada, el hold-out se abre una sola vez con
manifiesto firmado (Regla B), y las métricas primarias siguen siendo Calmar y Sortino
(constitución §2), con el Sharpe como secundario.

## Alternativas descartadas

- **Contador propio para la tesis.** Rechazada: crea dos verdades sobre el mismo activo y
  contradice explícitamente la línea 1604 del registro y §2 de la constitución.
- **Excluir 2025 del hold-out para «limpiarlo».** Rechazada: dejaría el hold-out en ~450
  sesiones, por debajo del umbral de 500 de §11.2, volviendo indecidible el contraste
  principal — se perdería más potencia de la que se gana en pureza, y el N heredado
  seguiría siendo 111 igualmente.
- **Reportar ambos DSR (con y sin herencia).** Rechazada: el «sin herencia» no tiene
  interpretación válida y su único efecto sería ofrecer una cifra más amable que citar.

## Verificación

```bash
python -m pytest tests/regression/test_trial_ledger.py \
                 tests/regression/test_hypothesis_registry_consistency.py \
                 tests/regression/test_selection_bias_metrics.py -q
python -c "import yaml; p=yaml.safe_load(open('config/research/partition.yaml')); \
           assert p['trials']['inherited_n'] == 111; print('herencia declarada OK')"
```

Ningún resultado de la tesis puede publicarse con un `n_trials` menor que el
`n_trials_total` del registro en el momento de calcularlo.
