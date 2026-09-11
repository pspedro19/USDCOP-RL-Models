---
kind: roadmap
status: PARTIAL
version: 3.0.0
last_verified: 2026-09-11
supersedes: [specs/planes/06-PRE-REGISTRATION.md]
code_anchors:
  - config/experiments/thesis_ppo_v2.yaml
  - config/research/feature_schema_v2.json
  - config/research/macro_availability.yaml
  - data/thesis/research_data_portable_v2.pkl
  - scripts/analysis/thesis_train_ppo.py
  - scripts/analysis/thesis_statistics.py
---

# Pre-registro v3 — EXP-TESIS-RL-02

Este documento gobierna la versión corregida después de la auditoría EXP-TESIS-RL-01. Está en
`PARTIAL` hasta que el operador lo firme antes de la primera corrida v2 de mercado. Los replays
de 2023 y 2024–2026 son diagnósticos retrospectivos y no pueden convertirse en evidencia
confirmatoria.

## Identidad congelada

| Elemento | Valor |
|---|---|
| Schema | `feature_schema_v2.json`, 37 features, SHA `a0568db4b953604cabb6d64eab73631419f65184d0fc46f35f36c695487484cb` |
| Dataset portable | `research_data_portable_v2.pkl`, identidad `c4d32158a08e37c735b44137efa9003ce0a1b7a7b2fd46f35f36c695487484cb` |
| Macro | `macro_availability.yaml`, disponibilidad estricta anterior a apertura |
| Semillas | `42, 123, 456, 789, 1337` |
| Anualización | 221 sesiones/año |

## Universo y cargos

El universo v2 contiene únicamente `ppo_regime_v2` y `ppo_backbone_v2`, cinco semillas cada
uno, más `always_flat`, B1, B1′, NULL-A, random y las reglas intradía fijas. Las correcciones,
replays retrospectivos y fixtures sintéticas cobran cero trials de mercado. El control shuffle
cobra un FT; cada configuración v2 evaluada en selección cobra un AT; cada brazo forward cobra
un AT. El ledger se actualiza antes de mirar el resultado.

## Hipótesis y juez

- H1′: una política PPO v2 supera `always_flat` en retorno diario neto pareado.
- H2: la información de régimen mejora a `ppo_backbone_v2`; solo se interpreta con potencia
  declarada y bootstrap jerárquico por semilla.
- Evaluación retrospectiva: desarrollo y selección, con retraso 0/1/2, surrogate intrasesión y
  escenarios de costo bajo/central/alto.
- Juez confirmatorio: únicamente forward después del freeze, una mirada a `n=120` sesiones,
  barras no selladas = NA.

## Regla de éxito

No se reclama edge salvo que el resultado por semilla, la mediana, el DSR (`N` del ledger), los
baselines, el stress ×2 y el retraso de una barra sean favorables. Si v2 pierde contra flat en
selección, la familia PPO se cierra y el forward queda como replicación descriptiva.

## Prohibiciones

- No elegir hiperparámetros mirando selección u hold-out.
- No usar macro del mismo día, cero-fill ni fallback de fuente.
- No reportar Sharpe/p con menos de 20 operaciones.
- No llamar confirmatorio a un replay retrospectivo.
- No ejecutar el juez forward antes de 120 sesiones ni abrir una segunda mirada.
