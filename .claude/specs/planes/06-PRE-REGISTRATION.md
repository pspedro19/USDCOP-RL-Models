---
kind: roadmap
status: IMPLEMENTED
contract: CTR-RESEARCH-PREREG-001
version: 2.0.0
last_verified: 2026-08-25
supersedes: []
code_anchors:
  - config/research/partition.yaml
  - .claude/specs/adr/ADR-0023-contador-trials-unico-tesis.md
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - services/common/metrics.py
  - src/evaluation/benchmarks.py
---

# PRE-REGISTRO DEL HOLD-OUT — tesis USD/COP (RL / LLM / Híbrido)

> **Regla B**: *no se abre el hold-out hasta haber completado F8 y firmado el manifiesto
> pre-hold-out.* Este documento **es** ese manifiesto. Todo lo de aquí se fija **antes** de
> ver un solo número del hold-out; cambiar cualquier línea después de abrirlo invalida la
> pretensión confirmatoria del trabajo.
>
> Documento maestro: [`06-tesis-rl-llm-hibrido.md`](06-tesis-rl-llm-hibrido.md).
> Partición congelada: `config/research/partition.yaml`.
> Contabilidad de trials: [`ADR-0023`](../adr/ADR-0023-contador-trials-unico-tesis.md).

**Estado de la firma**: `IMPLEMENTED` — **firmado el 2026-08-25** (§9). F8 completa: matriz OOF
congelada (10 corridas), PBO y DSR calculados, refit ejecutado, y White/SPA declarados NO
computables con su razón. El universo quedó cerrado por la Enmienda #1 **antes** de abrir.

---

## 1. La ventana, y por qué esta

| Bloque | Rango | Sesiones | Uso |
|---|---|---|---|
| Desarrollo | 2020-01-02 → 2022-12-31 | 777 | HPO (F5). Todo Optuna aquí y solo aquí |
| Selección | 2023-01-01 → 2023-12-31 | 260 | Top-3 (F5/F6/F7) |
| Refit | 2020-01-02 → 2023-12-31 | 1.037 | Una vez, antes de abrir (F8) |
| **HOLD-OUT** | **2024-01-01 → 2026-08-24** | **677** | **Una apertura (F8b)** |

El §1 del plan proponía elegir entre dos opciones que arrancaban en 2019. **No existe
2019**: la serie de 5 minutos empieza el 2020-01-02 (hay 9 sesiones sueltas de dic-2019,
insuficientes para formar bloque). Se adoptó la Opción B desplazada por el dato, más el
tramo de 2026 — la máxima potencia disponible. 677 supera el umbral de 500 que §11.2 llama
«el límite» para que el contraste principal sea decidible.

Hashes de congelación: `calendar_sha256_16 = c197f7c85c585dab`,
`data_sha256_16 = dc05e649a625906d`.

---

## 2. Universo cerrado

> ### ENMIENDA #1 (2026-08-25, PRE-apertura — 0 miradas al hold-out)
>
> El universo se **reduce a dos brazos**: `ppo_regime` y `ppo_backbone`. `llm_direct` y
> `hybrid` **no se construyeron** (recorte de alcance decidido por el operador), así que su
> grid de 16 celdas (β, τ) tampoco existe.
>
> **Esta enmienda se escribe ANTES de abrir el hold-out y sin haber mirado un solo dato suyo**,
> que es la única forma en que una enmienda a un pre-registro es legítima. Reducir el universo
> *después* de ver los resultados sería elegir contra quién compararse a posteriori.
>
> Reducir el universo **no** afloja ningún criterio:
>
> - **No baja el listón.** Los baselines obligatorios se mantienen íntegros, incluido
>   `always_flat`, que la Fase E demostró que es el listón real y no una formalidad.
> - **No mejora el DSR.** El deflactor es `n_trials` del ACTIVO (113), no el tamaño del
>   universo.
> - **Sí invalida White RC y SPA**, que contrastan contra el universo. Con dos candidatos no
>   tienen contenido; se declara la omisión (§11.6 del documento maestro) en vez de publicar el
>   número vacío.
> - **Sí reduce la familia de Holm a un miembro** (H2), de modo que no hay corrección por
>   multiplicidad. Es consecuencia del recorte, no una ventaja obtenida.
>
> Baselines que se mantienen exactamente como estaban: `b1_buy_hold` (pasivo **y** 1× intradía,
> los dos), `b1_matched` (B1′), `null_a`, `random`, `always_flat`.


Lo que se evalúa en el hold-out queda fijado aquí. **No se añade nada después.**

**Brazos**

| Id | Descripción | Rol |
|---|---|---|
| `ppo_regime` | PPO + régimen en el estado | Brazo 1 (§10.1) |
| `ppo_backbone` | PPO sin régimen, mismos hiperparámetros | Ablación controlada (§10.3) |
| `llm_direct` | LLM decisor directo | Brazo 2 (§10.4) |
| `hybrid` | Política = misma instancia congelada de `ppo_regime`; `s_d` solo en wrapper | Brazo 3 (§10.5) |

**Baselines y controles obligatorios** (constitución §3; sin ellos no hay PROMOTE)

| Id | Qué responde | Implementación |
|---|---|---|
| `b1_buy_hold` | ¿bate a estar largo 1×? | `src/evaluation/benchmarks.py::buy_and_hold` |
| `b1_matched` | ¿o solo tiene menos beta? | `scripts/analysis/exposure_matched.py` |
| `null_a` | ¿aporta algo sobre el baseline tonto del track? | patrón de `cop_null_suite.py` |
| `random` | ¿bate al azar? | `…::random_signals` |
| `always_flat` | ¿bate a no operar? | `…::always_flat` |
| `baseline_matched` | supervisado con la misma información (H1) | §10.6/§10.7 |

**Grid del híbrido**: 16 celdas distintas de (β, τ) según §10.5. Ni una más.

**Costos**: contrato §9.3, con stress ×1/×2/×3 (constitución §3.4). Si muere al doble, REJECT.

---

## 3. Hipótesis, con su estatus inferencial declarado ANTES

**Confirmatorias** — familia Holm–Bonferroni, FWER 0,05. Solo estas dos, porque solo estas
tienen `ρ` alta por construcción:

- **H2**: `ppo_regime` supera a `ppo_backbone` (ablación controlada).
- **H3**: `hybrid` supera al mismo `ppo_regime` del que parte.

**Reportada con precisión declarada, NO como contraste decidible:**

- **H1**: `ppo_regime` frente a `baseline_matched`. Se publica la diferencia pareada con su
  IC, `ρ` y `n`, **junto con la afirmación —hecha aquí, antes de mirar— de que el diseño no
  puede resolver este contraste**. La ausencia de significancia no será evidencia de
  equivalencia y no se redactará como tal.

**Exploratoria:**

- **H4**: `llm_direct` frente a `ppo_regime`, con las salvedades de §10.4 (contaminación del
  corpus: el modelo ya sabe cómo terminó la historia).

**Estadístico primario**: diferencia pareada de Sharpe anualizado sobre `retorno_diario_d`,
en los mismos días de la máscara común. Bootstrap estacionario pareado, 10.000 réplicas,
longitud de bloque por regla automática truncada a 5–20 sesiones.

**Métrica de graduación** (constitución §2): **Calmar**, y Sortino. El Sharpe es secundario
pese a ser el estadístico del contraste.

---

## 4. Regla de fallo del PBO — escrita antes de calcularlo

```text
Si PBO > 0.20:
  - NO se modifica el universo candidato ni se retunea nada;
  - se suspende la pretensión confirmatoria del trabajo;
  - el hold-out puede abrirse UNICAMENTE como evaluacion de un sistema con
    riesgo alto de seleccion, etiquetado asi en TODAS las tablas,
    o el estudio se cierra sin apertura;
  - la eleccion entre esas dos opciones se registra por escrito ANTES de
    calcular el PBO.
```

**Elección registrada ex-ante**: si PBO > 0,20 se **abre igualmente**, etiquetado como
evaluación de alto riesgo de selección en todas las tablas y en el resumen. Razón: el valor
docente y de registro de un resultado honesto con etiqueta supera al de no abrir, y cerrar
sin abrir dejaría el trabajo sin ningún dato confirmatorio a cuatro meses de la defensa.
Esta decisión está tomada **antes** de conocer el PBO y no se revisará después.

---

## 5. Deflated Sharpe: cantidad reportada, no compuerta

`N_trials` **no arranca en cero**. Per [`ADR-0023`](../adr/ADR-0023-contador-trials-unico-tesis.md):

- **111 trials heredados** de USD/COP (`HYPOTHESIS-REGISTRY.md:15`; la línea 1604: *«la
  partición jamás resetea N»*).
- Más los de esta tesis: 60 de Optuna (F5), las ablaciones factoriales, las 16 celdas del
  grid híbrido, los baselines y controles.

El DSR (`services/common/metrics.py::deflated_sharpe_ratio`) se recomputa **siempre** con
el total. Se publica el número con su regla de interpretación (§11.8), no un aprobado.

**Contaminación parcial declarada**: el hold-out incluye 2025, que el track H5 de
producción ya grid-searcheó (el «42-cell grid, #8 of 42» de `CLAUDE.md`). Se mantiene
dentro para no caer por debajo del umbral de potencia de 500 sesiones. **No se presentará
como período virgen en ninguna tabla ni en el texto.**

---

## 6. Máscara de evaluación común

Todos los sistemas se evalúan sobre **el mismo** conjunto de sesiones válidas (§9.5,
test 15). Una sesión inválida no entra como retorno 0 ni cuenta en `n`.

Se excluyen, por criterio fijado aquí:

- Los **11 festivos colombianos con sesión sintética completa** (60 barras en día de
  mercado cerrado = relleno del proveedor; lista congelada en
  `tests/regression/test_seed_ohlcv_integrity.py`).
- Los festivos con residuo parcial (2–18 barras).
- Las **288 sesiones incompletas** de 1.723 (menos de 60 barras).

La máscara se hashea y el hash se publica junto a los resultados.

---

## 7. Contabilidad económica (§9.2)

- Retornos **simples**, equity **compuesta**. No se mezclan log-retornos con costos simples.
- El entorno fuerza `w = 0` al cierre de sesión **y cobra su costo**, que entra en el
  retorno diario (test 13).
- `rf = 0`, justificado por materialidad.
- Con **N < 20 trades** se reportan solo conteo y PnL — ni Sharpe ni p-value
  (constitución §6, `test_small_sample_stats_suppressed.py`).

---

## 8. Qué se publica pase lo que pase

Compromiso adquirido antes de ver resultados:

1. La tabla de baselines (§10.6/§10.8) se publica **aunque ningún brazo la bata**.
2. Un contraste sin poder se reporta **como indecidible**, no como «sin diferencias
   significativas» (§11.1, §3.9).
3. Los resultados negativos son válidos (§3.6). Si `ppo_regime` no bate a `null_a`, la
   conclusión escrita será que **el baseline es la estrategia** (constitución §3).
4. El hold-out se abre **una vez**. Si se reabriera, la reapertura se declara en el texto
   con su motivo y el conteo de aperturas.

---

## 9. Firma

| Campo | Valor |
|---|---|
| Pre-registrado | 2026-08-24 |
| Enmienda #1 (universo a 2 brazos) | 2026-08-25, **pre-apertura, 0 miradas** |
| Partición congelada | `config/research/partition.yaml` (`CTR-RESEARCH-PARTITION-001`) |
| Máscara de evaluación | `f6958be1b59e9767` (584 sesiones efectivas en hold-out) |
| Esquema de features | `5427409c451bc46b` (39 features) |
| `n_trials` heredado | 111 |
| `n_trials` tras la apertura H-TESIS-RL-01 | **113** (+2 AT) |
| **Firmado (F8 completa)** | **2026-08-25** |
| Hold-out abierto | ver `outputs/thesis/holdout_opening.json` |

### Qué se completó para poder firmar

| Requisito de F8 | Estado | Evidencia |
|---|---|---|
| Universo cerrado | HECHO | Enmienda #1, escrita antes de abrir |
| Matriz OOF congelada | HECHO | 10 corridas en `data/thesis/ppo/*.json` |
| PBO | HECHO | 0.116 sobre 12.870 particiones (selección) |
| DSR | HECHO | 0.0000 con `n_trials=111` |
| White RC / SPA | **NO COMPUTADOS, declarado** | universo de 2 candidatos; §11.6 |
| Refit desarrollo+selección | HECHO | `research_thesis_ppo_training` con `refit: true` |
| Regla de fallo del PBO escrita antes | HECHO | §4 de este documento |
| Análisis de poder | HECHO | mínimo detectable ΔSharpe ≈ 0.7 al 80%, §11.2 |

### Lo que YA se sabe al firmar, y por qué se abre igual

La firma NO se emite sobre un resultado desconocido. El bloque de **selección** ya dio un
veredicto decidible y **negativo**: `ppo_regime` queda por debajo de `always_flat` con
ΔSharpe −4.640 (p < 0.0001), 0 de 10 semillas positivas, DSR 0.0000, y muerte al doble de
costos.

Conviene decir explícitamente por qué se abre el hold-out sabiendo esto, porque hay dos
lecturas y solo una es legítima:

- **NO se abre para buscar un resultado mejor.** Eso sería usar el hold-out como segundo
  bloque de selección, que es exactamente lo que la Regla B impide. El veredicto ya está
  tomado sobre selección y **no cambiará** con lo que salga del hold-out.
- **Se abre porque estaba pre-registrado y porque el hold-out es el bloque con potencia**
  (584 sesiones efectivas frente a 234). Es donde la magnitud del rechazo se mide bien, y
  es el único período que incluye 2024-2026.

Compromiso adicional que se firma aquí: **si el hold-out saliera positivo**, eso NO
revierte el rechazo — se reportaría como una discrepancia entre bloques que exige
explicación, no como una absolución. Un resultado que solo aparece en el bloque que se abre
al final, tras un rechazo decidido en selección, es precisamente el patrón que el DSR y el
PBO existen para desconfiar.

Mientras la fila «Firmado» estuvo vacía, `status` permaneció en `PARTIAL` y el gate de código
(`thesis_open_holdout.py`, `thesis_statistics.py --block holdout`) devolvió 2 sin leer datos.
