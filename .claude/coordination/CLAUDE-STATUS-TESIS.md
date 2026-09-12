# CLAUDE-STATUS — carril tesis RL (BL-50)

> Fichero corto a proposito. `CLAUDE-STATUS.md` tiene 190 000 lineas y nadie lo lee; este
> cubre solo el carril de la tesis y se sobrescribe entero en cada latido.

**Ultimo latido**: 2026-09-11T20:35-05:00

## Lo que estoy corriendo
- `outputs/thesis-repair/ppo_v2_recipe_flat/` — 10 corridas (2 configs x 5 semillas, 300k pasos),
  dataset v2, receta `flat_init_no_turn` aplicada de verdad. **Diagnostico retrospectivo, 0 trials.**

## Lo que NO toco
- `outputs/thesis-repair/ppo_v2_diagnostic_full/` (brazo de control, lo corre Codex)
- `docs/analysis/**`, `.claude/rules/**`, `HYPOTHESIS-REGISTRY.md`, `CODEX-STATUS.md`
- `.env` (ni leer ni publicar) · la firma del prerregistro

## Bloqueos reales
1. **DeepSeek/Azure**: `run_thesis_llm.py:144` exige `status: SIGNED` en el prereg v3. Solo el
   operador firma. Todo lo demas del brazo LLM esta listo (13.334 contextos exportados, config
   congelado con `temperature 0.1`, `top_p 0.9`, `max_tokens 256`, `prompt_version thesis-llm-trader-v1`).
2. **Hibrido**: sin definir ex-ante (PPO+FinMA-ES **o** PPO+LLM, no ambos).
