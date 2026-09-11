# Brief para el operador — contabilidad de trials del programa de reparación (BL-50)

> **Qué es esto:** el borrador de las dos entradas que hay que aplicar al registro de
> hipótesis de USD/COP y al ledger. **Claude no edita `HYPOTHESIS-REGISTRY.md` ni
> `registries/ledger.jsonl`**; los redacta y el operador los aplica con su proceso, que es
> quien firma. Fecha: 2026-09-11 · Contrato: `CTR-QUANT-CONSTITUTION-001`.

## Estado de partida (verificado hoy)

`scripts/validation/check_trial_ledger.py` → `OK`: 243 trials globales (FT = 55, AT = 188),
de los cuales **usdcop = 115**, que es exactamente lo que declara el front matter del
registro. Las sumas por activo, la unicidad, la cadena de hashes y las familias cuadran.

## Entrada 1 — CORRIGENDUM H-TESIS-RL-01 (0 trials)

**No cobra trials.** Corrige el reporte de una hipótesis ya cerrada; no mira ningún resultado
nuevo ni selecciona nada.

- El rechazo económico se mantiene: neto negativo frente a `always_flat` en los dos bloques,
  las diez corridas netas negativas, muerte al doble de costos (`ppo_regime` ×2 = −84,5866 %).
- Se retira la afirmación «el agente aprende una política con señal real». Motivo comprobado:
  fuga de disponibilidad temporal en las tres features macro (`causality_gate_pass = False`
  en el diagnóstico del 2026-09-10), más convenciones de reporte que inflaban el bruto.
- Se retira «10/10 semillas con bruto positivo»: en compuesto, `ppo_backbone` semilla 456 da
  **−0,0211 %**.
- Se retira la caída de ρ 0,97 → 0,58: los valores son **0,5853** (selección) y **0,5821**
  (hold-out), y la potencia era insuficiente en ambos bloques.
- El conteo `N ≥ 125` que aparecía en la revisión Claude **no está demostrado**; el número
  vigente sigue siendo **115** hasta que se concilien cadencias y sensibilidades.

**Acción:** añadir la nota de corrigendum al registro citando
`docs/analysis/exp-tesis-rl-01-auditoria-2026-09-10.md` y el corrigendum de
`.claude/specs/planes/06-RESULTADOS.md` (v1.1.0). **`n_trials_total` no cambia: 115.**

## Entrada 2 — APERTURA H-TESIS-RL-02 (v2, reparada)

Se abre cuando el pre-registro v3 esté **firmado**, no antes. Cargos previstos:

| Ítem | Cargo | Cuándo se devenga |
|---|---|---|
| Reparaciones de datos y entorno, re-mediciones contables, replays retrospectivos etiquetados | **0** | — |
| Sanidad sintética S1-S4 y sondas ordenadas | **0** | no tocan datos de mercado |
| Control shuffle sobre desarrollo | **+1 FT** | al leer su resultado |
| `ppo_regime_v2` y `ppo_backbone_v2` evaluados en selección | **+2 AT** | al mirar selección |
| Baselines intradía (reglas fijas, dos reglas por régimen, horizonte) | **+1 AT c/u** | al mirar su resultado |
| Brazos forward v2 | **+2 AT** | en la primera liquidación mirada |

**Total previsto:** 115 → **118** antes del forward v2; 120 con él, más los baselines que se
ejecuten. `n_trials_total` del front matter y el bloque `<!-- LEDGER-TOTALS -->` de
`registries/README.md` se actualizan **a la vez**, y `check_trial_ledger.py` debe seguir en
exit 0.

**Familia:** `registries/families/usdcop_rl_intraday_v2.yaml`, con `hypothesis_key` heredada
de v1 y `sibling_families: [usdcop_rl_intraday]` (y la arista inversa en v1), para que la
deflación sea de clúster y v2 no estrene un contador limpio. Eso es justamente lo que la
constitución §2 impide.

## La regla que no se negocia

Selección 2023 y hold-out 2024-26 **ya se miraron y motivaron estas correcciones**. Cualquier
evaluación de v2 sobre esos bloques es **diagnóstico retrospectivo** y así debe etiquetarse en
todo artefacto. El juez confirmatorio de v2 es el carril **forward**, desde su fecha de
congelación, con la mirada única declarada en v3.

## Qué necesita el operador decidir

1. Firmar el pre-registro v3 (hoy `status: PARTIAL`) antes de la primera corrida v2.
2. Ratificar o rebajar el **+1 FT** del control shuffle (es un control nulo; lo cargo por
   conservadurismo).
3. Identidad macro definitiva para el carril de investigación: Brent spot FRED de punta a
   punta, DGS2 desde FRED, DXY sin fallback a Fed Broad.
