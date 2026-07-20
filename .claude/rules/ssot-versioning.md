---
kind: rule
status: IMPLEMENTED
contract: CTR-SSOT-LIFECYCLE-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - config/pipeline_ssot.yaml
  - config/execution/smart_simple_v1.yaml
---
# Rule: Versionado SSOT

> **SSOT de las invariantes de config.** Estructura, campos `_meta`, ciclo de vida y cómo crear
> un experimento: `../specs/platform/ssot-lifecycle.md`.

## Invariantes

1. **Un experimento = un archivo SSOT completo y autónomo.** No un diff, no un override parcial,
   no anchors YAML. Razón: reproducibilidad — cualquiera re-corre el experimento con solo el config.
2. **El config se CONGELA cuando arranca el training.** Si hace falta ajustarlo, se ABORTA y se
   abre un experimento nuevo con otro ID.
3. **`v215b_baseline.yaml` es sagrado** — nunca se modifica. Todo experimento deriva de él
   copiando y cambiando UNA variable. Un baseline nuevo es un archivo nuevo.
4. **Nombre = `{experiment_id_lowercase}.yaml`** y `_meta.experiment_id` debe coincidir con
   `EXPERIMENT_QUEUE.md`.
5. **`_meta` es obligatorio** con: `experiment_id`, `contract_id`, `based_on`, `variable_changed`,
   `hypothesis`, `status`, `results`.
6. **`pipeline_ssot.yaml` es una COPIA del experimento activo**; cuando uno gana, su config pasa
   a serlo.

## DO NOT

- Do NOT modificar `v215b_baseline.yaml`.
- Do NOT modificar un config después de que arrancó el training — abre otro experimento.
- Do NOT usar diffs, overrides parciales ni anchors YAML como "config de experimento".
- Do NOT cambiar más de una variable por experimento (ver `experiment-protocol.md`).
- Do NOT dejar `_meta.results` sin poblar tras L4.
