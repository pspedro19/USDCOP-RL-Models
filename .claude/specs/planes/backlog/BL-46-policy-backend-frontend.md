---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/production/ForecastingBacktestSection.tsx
  - usdcop-trading-dashboard/components/gm/views/ProductionView.tsx
  - src/contracts/strategy_schema.py
---

# BL-46 — Políticas: backend (policy_version/signal) + frontend schema-driven (R4-R5)

**Fuente**: planes/05-rule-based-strategies.md §8-§9, §13 R4-R5 · **Ola**: 3-4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El frontend ya es motor-agnóstico para bundles (verificado: renderiza SPX-MA200 igual que COP-ML) pero NO hay panel explicativo por motor, ni rule_trace, ni policy_version en DB; los params por usuario (sb_trading_configs) se renderizan con campos fijos.

## Qué falta exactamente
R4: control.policy_version (engine_type, hashes, manifest_uri) + action.strategy_signal normalizada con decision_components JSONB + rule_trace_uri (converge con BL-42); endpoints GET /strategies*, POST /strategies/validate; librería de evaluación empaquetada (la MISMA de Airflow). R5: renderer único StrategyEngineExplanation con variantes RuleTracePanel/MLExplanationPanel/RLPolicyPanel/CompositeDecisionPanel; el frontend renderiza el trace, JAMÁS re-evalúa; sección presentation: con presentation_hash propio (cambiar etiqueta ≠ versión económica); configs por usuario renderizadas desde schema.

## Impacto frontend
Panel de explicación por motor en replay/production; tabla Condición|Observado|Umbral|Resultado. Conecta con BL-20 (la vista admin de interpretabilidad consume el mismo trace para rule-based).

## Dependencias
BL-45, BL-42 (misma tabla de señal — implementar juntas), BL-22.

## Verificación
RuleTracePanel muestra el trace real de MA200 sin recalcular (test: mock trace ⇒ render exacto); rbac de endpoints nuevos en la matriz.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando:   npx vitest run tests/unit/components/StrategyEngineExplanation.test.tsx
           (desde usdcop-trading-dashboard/)
verde:     5 passed

comando-2: python -m pytest tests/unit/test_policy_backend_contract.py -q
verde-2:   128 passed

comando-3: npx vitest run tests/unit/contracts/policy-backend-parity.test.ts
verde-3:   107 passed

muta:      StrategyEngineExplanation — re-evaluar la condición EN REACT en vez de renderizar
           el veredicto que el backend selló en el rule_trace
espera:    2 failed, uno de ellos con TRAZA CONTRADICTORIA (el panel afirma un resultado
           distinto al que trae el trace) — es la invariante 7 de
           .claude/rules/strategy-engines.md: "el frontend renderiza el rule_trace, NUNCA
           re-evalúa condiciones"
```

**Historial honesto**: BL-46 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`) — no hubo defecto que cerrar el 2026-07-28. Lo que lo hace fiable no es que caiga
un test, sino **cómo** cae: el segundo rojo es una **contradicción interna visible** —el panel
publica un veredicto distinto del que el backend selló—, así que la mutación no se puede
"arreglar" ajustando el fixture; habría que hacer que el frontend mienta consistentemente
sobre su propia fuente. Es la diferencia entre proteger el render y proteger la invariante.

## Notas constitución
El frontend presenta hechos; no decide ni recalcula (ley 12 FABRIC).

## Brecha registrada: `PolicyVersionRecord` existe, valida estricto y NO TIENE PRODUCTORES

**Registro documental** (CLAUDE, 2026-08-06, autorizado en CXD-623). No se implementa
productor ni se toca el `status` de esta ficha: eso requiere propuesta y review propios.

### El hecho, medido

`src/contracts/policy_version.py` declara `PolicyVersionRecord` con **`feature_set_hash: str`
obligatorio**, y su `__post_init__` le aplica `require_hash` — patrón `^sha256:[0-9a-f]{64}$`,
así que un `null` **levanta**. Junto a él, `params_hash`, `policy_hash` y `resample_policy_hash`
reciben el mismo trato.

Búsqueda de constructores en todo el repo (`rg "PolicyVersionRecord"`, excluyendo `.tmp/`):

| Anchor | Qué hace |
|---|---|
| `src/contracts/policy_version.py` | define la dataclass y valida |
| `src/policy_engine/runner.py:39` | la **importa** |
| `src/policy_engine/runner.py:284` | `write_policy_version_index(records: Iterable[PolicyVersionRecord], …)` — la **tipa** |
| `src/policy_engine/runner.py:300` | `isinstance(record, PolicyVersionRecord)` — la **valida** |
| `tests/unit/test_policy_backend_contract.py:33,96` | `from_dict` en tests |
| **constructores productivos** | **CERO** |

### Por qué importa, y qué NO significa

El índice `control.policy_version` que la API del dashboard lee (`GET /api/strategies`) se
escribe con `write_policy_version_index`, que exige registros validados. **Nadie construye
uno**, así que el índice nunca se ha producido por esta vía.

Esto explica un detalle que parecía otra cosa: los cuatro specs traían
`governance.feature_set_hash: null` y **eso no rompía nada** — no porque el `null` fuera
aceptable, sino porque **ningún productor lo leía jamás**. Un campo obligatorio que nunca se
puebla porque nunca se construye el objeto que lo exige.

Es el mismo patrón que ya apareció dos veces en esta serie y conviene nombrarlo como patrón:
`resolve_feature_snapshot` tenía cero llamadores productivos antes de C-010, y `status_ceiling`
fue escrita fail-closed y no la consultaba nadie hasta CXD-622. **Mecanismo correcto, sin
llamador** — no falla, no aparece en ninguna suite roja, y da la impresión de estar cubierto
porque el código existe y está bien escrito.

**Lo que NO se afirma aquí**: que el contrato esté mal diseñado, ni que haya que construir el
productor ya. Puede que el índice deba producirse en otro punto del ciclo, o que
`control.policy_version` llegue con la migración de `fabric-v1`. Decidirlo es alcance de esta
ficha, no de un registro.

### Qué haría falta para cerrarla (no ejecutado)

1. Decidir **quién** escribe el índice y **cuándo** (¿tras el freeze de una policy?, ¿en el
   deploy?, ¿en el DAG?).
2. Poblar los cuatro hashes que el record exige. `policy_hash` ya existe en los specs;
   `feature_set_hash` existe ahora **sólo en `spx500_daily_ma200_v1`** (piloto `91400773`,
   deuda declarada para las otras tres); `params_hash` y `resample_policy_hash` **no existen
   en ningún spec** — hay que decidir de dónde salen antes de prometer el record.
3. Un candado que falle si el índice se declara producido y no lo está, para que esta brecha
   no pueda volver en silencio.

**BL-46 sigue `PARTIAL`.** Este registro no mueve su estado ni añade alcance.
