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
| 3. Sanidad del optimizador | S1 ejecutada con las 5 semillas y las 4 sondas: **ninguna receta pasa**. Faltan S2-S4, que sólo tienen sentido con una receta candidata | HECHO en su parte decisiva |
| 4. Reentreno sin fugas y juez forward | **BLOQUEADA por la compuerta de sanidad**, no por falta de tiempo: el v3 exige congelar una receta que pase S1 y no hay ninguna | BLOQUEADA |
| 5. Corrigendum | Correcciones de §0-§7 con cifras de la evidencia | HECHO |
| 6. Gobernanza | Brief de contabilidad redactado para el operador; ledger y registro sin tocar desde este carril | PARCIAL |

## Hallazgos de la ejecución (2026-09-11)

Tres defectos que no estaban en el plan y salieron al ejecutarlo. Los tres comparten forma:
**código correcto, committeado y nunca ejecutado de verdad.**

1. **El runner de sanidad no arrancaba.** `thesis_ppo_sanity.py` se invoca como fichero y le
   faltaba la raíz del repo en `sys.path`: moría en el primer `import`. Estaba committeado
   desde el día anterior. Corregido.
2. **El pre-registro congelaba un hash que no era el del artefacto.** La fila del schema
   terminaba en `…46f35f36c695487484cb`, que es la cola de la identidad del dataset portable:
   una pegada defectuosa había sustituido los últimos 18 caracteres. Un congelamiento cuyo
   hash no se puede verificar no congela nada. Corregido y cableado en
   `tests/regression/test_prereg_v3_identity.py`, que además prohíbe que dos hashes de la
   tabla compartan cola.
3. **La semilla 2024 en lugar de 1337**, contra la regla 2 del protocolo ("sin excepciones"),
   contra la identidad publicada en el propio pre-registro y contra `thesis_train_ppo.py`.
   La evidencia de sanidad no era comparable por semilla con las corridas que justificaba.
   Corregido y cableado por AST en `test_thesis_runners_use_protocol_seeds.py`.

Y uno de datos, fuera del alcance de la tesis pero encontrado al cerrar su hueco de entrada:
**escribir un seed era una sobrescritura, no un UPSERT**, así que el comando documentado de
ingesta borraba seis años de USD/COP después de dar `PASS`. Corregido en
`scripts/data/ingest_asset_ohlcv.py` con cinco tests y mutación verificada.

## Estado de los datos de entrada

El seed intradía de USD/COP se refrescó al **2026-09-10** (99.714 → 100.674 filas, 960 barras
nuevas) y `fx_multi` quedó alineado respetando su esquema `decimal128(11,6)`. El juez forward
vuelve a tener datos antes del corte del 16 de septiembre.

**Brecha declarada:** el ledger de paper 2026 no se puede regenerar sin Postgres, porque el
linaje BL-24 se persiste en la misma transacción que escribe el fichero. El puerto 5432 lo
ocupa otro proyecto del operador. El número reproducible hoy es **+0,66 % con 12 operaciones**
para `smart_simple_v11`; el artefacto publicado dice +2,90 % y está desfasado.

**Segunda brecha, destapada por la primera:** refrescar el precio dejó a la vista que
`MACRO_DAILY_CLEAN` sigue en el **2026-08-24**, trece días hábiles por detrás. Con la nueva
regla de disponibilidad —que es correcta— el carril live **no puede sellar** ninguna de esas
doce sesiones, así que el juez forward se queda sin datos aunque el precio esté fresco. Antes
del refresco esto era invisible: precio y macro estaban igual de viejos y nada se quejaba.
`tests/regression/test_live_spec_parity.py::test_macro_is_not_behind_price_for_the_live_lane`
lo deja **rojo a propósito** con la instrucción. No se rellena a mano: mezclar Brent spot con
futuros, o DGS2 de dos fuentes, es exactamente lo que la auditoría acaba de limpiar.

## Qué significa "forward" en la serie 2026 de v11 (2026-09-11)

Importa para cualquier afirmación comercial, así que se deja escrito antes de que lo pregunte
un tercero. El generador etiqueta la serie 2026 de `smart_simple_v11` como *"forward real todo
2026 (producción)"*. Eso es cierto en el sentido **metodológico**: la estrategia está congelada
desde marzo de 2026 con fecha y hashes verificables, así que el resultado de 2026 no pudo
elegirse mirando el período.

No es cierto en el sentido de **libro sellado semana a semana**. Las tablas de ejecución
(`forecast_h5_paper_trading`, `forecast_h5_executions`) tienen **ocho filas, todas con el mismo
`created_at` del 2026-07-05**: son un relleno retroactivo, no una captura semanal, y cubren
hasta junio mientras el ledger reporta doce operaciones. El rendimiento de 2026 se obtiene
**reejecutando la regla congelada** sobre los datos posteriores, y esa reejecución sí es
reproducible por un tercero.

La distinción es vendible tal cual —"regla congelada, resultado reproducible"— y no lo sería
como "track record auditado en vivo". Construir el sellado semanal real es trabajo de producto,
y es lo que el carril forward de la tesis (BL-50 etapa 4.3) hace bien por diseño.

## Coste medido de la Etapa 3 y cómo reanudarla (2026-09-11)

Calibrado en esta máquina: **20.000 pasos = 147 s**, o sea **~12,3 min por semilla** a los
100.000 pasos declarados. De ahí:

| Alcance | Corridas | Tiempo |
|---|---:|---:|
| S1 base (5 semillas) | 5 | ~61 min |
| S1+S4 base (ruta mínima del plan) | 10 | ~2 h |
| Protocolo completo con sondas | hasta 100 | ~20 h |

**El entorno corta los trabajos de fondo por presión de memoria**: Docker/WSL retiene ~6,5 GB
para otro proyecto del operador y quedan ~4-5 GB, mientras cada corrida PPO pide ~1,4 GB. Por
eso la Etapa 3 se ejecuta **una semilla por proceso**, escribiendo
`outputs/thesis-repair/sanity/S1_seed<N>.json`: el bucle salta las semillas ya hechas, así que
cada corte conserva lo avanzado y basta relanzar.

```bash
for s in 42 123 456 789 1337; do
  out="outputs/thesis-repair/sanity/S1_seed${s}.json"
  [ -f "$out" ] && continue
  OMP_NUM_THREADS=1 python scripts/analysis/thesis_ppo_sanity.py \
      --fixture S1 --seed "$s" --timesteps 100000 --output "$out"
done
```

**Aviso operativo:** cada corte deja el proceso vivo (mata el shell, no el árbol), y la
memoria del huérfano provoca el corte siguiente. Antes de relanzar hay que barrerlos; se
encontraron tres en una noche.

**Resultado de S1 a 100.000 pasos (2026-09-11): ninguna receta pasa.**

| Receta | Planas | Exposición media por semilla |
|---|---:|---|
| baseline | 0/5 | 0,527 · 0,966 · 0,485 · 0,985 · 0,968 |
| `ent_coef = 0` | 0/2 | 0,958 · 0,965 |
| `norm_reward = False` | 0/2 | 0,976 · 0,963 |
| `γ = 1.0` | 0/2 | 0,603 · 0,618 |
| `κ_turn = 1.0` | 1/3 | **0,052** · 0,499 · 0,580 |

Una sonda se descarta al segundo fallo, porque ya no puede llegar a 4/5. Quitar entropía o
normalización **empeora** el churn (de 0,53 a ~0,97), así que no eran la causa. La única que
produjo una semilla plana es `κ_turn`, el penalizador de turnover que §9.6 del diseño
especificaba y la tesis nunca implementó.

**Esto reordena el plan.** La Etapa 4 no está pendiente por tiempo: está **bloqueada por la
compuerta**. Reentrenar v2 con una receta que no encuentra el flat sobre ruido produciría otra
conclusión confundida entre optimizador y mercado, que es el defecto que este programa existe
para corregir. La búsqueda de receta sigue en terreno sintético, donde no se gasta ningún trial.

Direcciones que el propio experimento sugiere, ninguna ejecutada: `κ_turn` por encima de 1
—el único eje que movió la aguja—, presupuesto mayor que 100.000 pasos (≈3,4 pasadas sobre 500
sesiones sintéticas), y revisar si el reward ×100 con `clip_reward=10` recorta justo la señal
de costo. Cada una es una variable y se declara antes de mirarla.

**Detalle histórico:** Exposiciones medias 0,527 /
0,966 / 0,485 / 0,985 / 0,968 sobre una serie de ruido iid con costo, donde la política óptima
es no operar. La regla pre-registrada exigía ≥4/5 planas. Evidencia en
`outputs/thesis-repair/sanity_S1_protocol.json`; lectura en el corrigendum §6.

Se resolvió el problema de entorno que lo bloqueaba: `thesis_ppo_sanity.py` ahora **guarda y
restaura `VecNormalize` junto al checkpoint**, así que el entrenamiento se parte en tramos sin
que las estadísticas de normalización del reward se reinicien a mitad. Con eso cada semilla
cabe en dos llamadas de ~5 min. **Ojo con la semántica**: al reanudar, `--timesteps` es el
total acumulado, no el incremento.

## Siguiente acción del operador, en orden

1. `git push origin HEAD:refs/heads/main` — hay commits locales sin publicar y el push falla
   en sesión no interactiva por falta de credencial guardada.
2. Levantar el stack y correr el pipeline macro L0 hasta hoy. Desbloquea el sellado del
   carril live y el ledger de paper, en ese orden.
3. Firmar el pre-registro v3 (hoy `PARTIAL`) antes de la primera corrida v2.

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
