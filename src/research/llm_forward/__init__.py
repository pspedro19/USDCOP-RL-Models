"""Rama comparativa forward: RL congelado frente a LLM, sobre las mismas sesiones.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## Por qué existe

La tesis (`H-TESIS-RL-01`) se cerró con un rechazo y su hold-out se abrió una sola vez. El
brazo LLM del plan original (§10.4) **nunca fue recuperable**, y no por falta de esfuerzo:

- **No hay corpus histórico.** 181 artículos en 12 días distintos, todos posteriores al
  2026-07-05, para una partición que necesita cubrir 1.383 sesiones desde 2020.
- **Y aunque lo hubiera**, un LLM que hoy lee una noticia de 2021 ya sabe cómo terminó 2021.
  Ese conocimiento está en los pesos; ningún filtro de corpus lo quita. El diseño
  retrospectivo está roto de origen.

La única evaluación limpia de un LLM aquí es **prospectiva**, y vale exactamente lo que valga
su prueba de que la decisión existió antes del resultado. De ahí este paquete.

## Qué lo hace comparativo

No se puede comparar un brazo medido sobre 584 sesiones pasadas contra uno que empieza hoy con
cero. Así que los dos corren **en paralelo, sobre las mismas sesiones**, con:

- el **mismo contrato de costos** (`src/research/cost_model.py`, §9.3 de la tesis),
- el **mismo motor de liquidación** (`src/research/session_env.py::run_session`),
- y **`always_flat` como listón**, que es quien batió a todo en el hold-out.

| Brazo | Decisiones/sesión | Sella | Papel |
|---|---|---|---|
| `llm_direct_fwd_v1` | 1 | 07:15 COT | el brazo nuevo |
| `ppo_regime_fwd_k59` | 1 | 08:00 COT (barra 0) | comparable cabeza a cabeza |
| `ppo_regime_fwd_k1` | 59 | 08:00 COT | evidencia forward del brazo que la tesis evaluó |
| `always_flat` | 0 | — | el listón |

**Asimetría declarada**: los dos sellan antes de que exista `r_1`, así que la comparación es
causalmente limpia — pero **el RL ve la barra 0 y el LLM no**. Es una ventaja de información
del RL y va en toda tabla, no en una nota al pie.

## Lo que este paquete NO puede producir

**No rescata H1 ni H2.** El hold-out se abrió y el PBO quedó en 0,211 > 0,20. Un brazo que
arranca hoy es un experimento nuevo, y `config/research/preregistration_forward.yaml` lo fija
en `status: exploratory` para que el fichero mismo impida la pretensión confirmatoria después.

**No da potencia.** ~250 sesiones al año; de aquí a diciembre son ~85. Sabiendo que en el
hold-out ni 584 sesiones resolvieron H2, la potencia es despreciable. Esto es **un piloto de
factibilidad y un artefacto de ingeniería, no un test**.

**No resuelve el corpus.** Prospectivo elimina la fuga; no consigue documentos. Si las fuentes
colombianas no publican material accionable antes de las 08:00 COT, el brazo forward tiene el
mismo problema que el retrospectivo, descubierto tres meses más tarde. Por eso el inventario de
fuentes es la compuerta y va **antes** de gastar un token.
"""

from .canonical import chain_hash, sha256_text
from .corpus import CutoffViolation, RawDoc, filter_by_cutoff, session_cutoff_utc
from .ledger import Ledger, LedgerError
from .schema import CorpusDoc, Decision, DecisionRecord, LlmUsage, SettlementRecord

__all__ = [
    "chain_hash", "sha256_text",
    "CutoffViolation", "RawDoc", "filter_by_cutoff", "session_cutoff_utc",
    "Ledger", "LedgerError",
    "CorpusDoc", "Decision", "DecisionRecord", "LlmUsage", "SettlementRecord",
]
