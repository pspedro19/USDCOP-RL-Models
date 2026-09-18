---
kind: runbook
status: ready_for_execution
version: 1.0.0
last_verified: 2026-09-15
supersedes: []
code_anchors:
  - scripts/analysis/export_thesis_llm_contexts.py
  - scripts/analysis/run_thesis_llm.py
  - scripts/analysis/check_confirmatory_artifacts.py
---

# Runbook de ejecución LLM v4

Este runbook prepara la llamada de los proveedores sin leer ni guardar secretos en el
repositorio. Las credenciales deben estar expuestas por el proceso operador mediante las
variables declaradas en `config/research/llm_thesis.yaml`. No se permite fallback entre
proveedores ni seleccionar retrospectivamente el proveedor con mejor resultado.

## Artefacto ya preparado

El bundle forward contiene 162 sesiones × 59 decisiones = 9.558 contextos, todos con:

- portable SHA-256 `65534cc4984cd2bb52bd7f42b845fcd5cb7be214123d154b5b8bf6c5ef2eb585`;
- `dataset_block: forward`;
- `retrospective: false`;
- corte de información hasta el cierre de la barra anterior;
- prompt `thesis-llm-trader-v1`, temperatura 0,10, top-p 0,90 y 256 tokens.

El gate del bundle es:

```powershell
python scripts/analysis/check_confirmatory_artifacts.py `
  --training outputs/thesis-repair/confirmatory_v4_ppo_threads1 `
  --holdout outputs/thesis-repair/confirmatory_v4_stable/ppo_holdout_2024_2025_stable.json `
  --statistics outputs/thesis-repair/confirmatory_v4_stable/ppo_confirmatory_statistics_v4_stable.json `
  --portable outputs/thesis-repair/confirmatory_v4_stable/research_data_portable_v4_stable.pkl `
  --forward outputs/thesis-repair/confirmatory_v4_stable/ppo_forward_2026_partial.json `
  --forward-baselines outputs/thesis-repair/confirmatory_v4_stable/forward_baselines_2026.json `
  --forward-llm-contexts outputs/thesis-repair/confirmatory_v4_stable/llm_forward_contexts_2026.jsonl
```

## Ejecución por proveedor

También existe un orquestador fail-closed que verifica ambos proveedores antes de iniciar
el primero y luego valida, liquida y construye ambos híbridos:

```powershell
python scripts/analysis/run_forward_llm_e2e.py `
  --deepseek-model <DEEPSEEK_MODEL_CONGELADO> `
  --azure-model <AZURE_DEPLOYMENT_CONGELADO>
```

Para comprobar preparación sin llamadas:

```powershell
python scripts/analysis/run_forward_llm_e2e.py `
  --deepseek-model deepseek-chat `
  --azure-model gpt-4o-mini `
  --check-only
```

El identificador de modelo/deployment debe escribirse explícitamente en el ledger. Se
ejecuta un proveedor por vez, con ledgers distintos y sin `provider_fallback`:

```powershell
python scripts/analysis/run_thesis_llm.py `
  --input-jsonl outputs/thesis-repair/confirmatory_v4_stable/llm_forward_contexts_2026.jsonl `
  --provider deepseek `
  --model-id <DEEPSEEK_MODEL_CONGELADO> `
  --ledger data/thesis/llm/decisions_deepseek_forward_2026.jsonl `
  --portable outputs/thesis-repair/confirmatory_v4_stable/research_data_portable_v4_stable.pkl `
  --execute
```

```powershell
python scripts/analysis/run_thesis_llm.py `
  --input-jsonl outputs/thesis-repair/confirmatory_v4_stable/llm_forward_contexts_2026.jsonl `
  --provider azure_openai `
  --model-id <AZURE_DEPLOYMENT_CONGELADO> `
  --ledger data/thesis/llm/decisions_azure_forward_2026.jsonl `
  --portable outputs/thesis-repair/confirmatory_v4_stable/research_data_portable_v4_stable.pkl `
  --execute
```

El operador puede usar `--load-dotenv` únicamente en su entorno controlado; el runner
no imprime valores secretos. Antes de liquidar se debe validar cada ledger, exigir 9.558
decisiones únicas, cero respuestas no disponibles y cero decisiones inválidas, y registrar
el hash del prompt y de la respuesta cruda. Una interrupción se reanuda solo con
`--resume`; nunca se sobrescribe un ledger append-only.

La liquidación posterior usa los mismos specs forward:

```powershell
python scripts/analysis/settle_thesis_llm.py `
  --ledger data/thesis/llm/decisions_deepseek_forward_2026.jsonl `
  --block forward `
  --portable outputs/thesis-repair/confirmatory_v4_stable/research_data_portable_v4_stable.pkl `
  --forward-specs outputs/thesis-repair/confirmatory_v4_stable/forward_specs_2026.pkl `
  --strict-ledger `
  --output outputs/thesis-repair/confirmatory_v4_stable/llm_deepseek_forward_2026.json
```

Para el híbrido se usa `ppo_forward_actions_2026.json`, generado a partir de las cinco
semillas y agregado por mediana componente a componente; la regla de acuerdo/veto no se
puede cambiar después de observar el ledger:

```powershell
python scripts/analysis/thesis_hybrid.py `
  --ppo-weights outputs/thesis-repair/confirmatory_v4_stable/ppo_forward_actions_2026.json `
  --ppo-config ppo_regime `
  --ledger data/thesis/llm/decisions_deepseek_forward_2026.jsonl `
  --block forward `
  --portable outputs/thesis-repair/confirmatory_v4_stable/research_data_portable_v4_stable.pkl `
  --forward-specs outputs/thesis-repair/confirmatory_v4_stable/forward_specs_2026.pkl `
  --output outputs/thesis-repair/confirmatory_v4_stable/hybrid_deepseek_forward_2026.json
```

## Qué se puede publicar

Hasta que ambos ledgers hayan pasado esa validación y el preregistro del brazo esté
firmado, los resultados LLM son `pending`, no confirmatorios. El PPO y los baselines
forward ya están liquidados; sus resultados no dependen de las llamadas LLM. Un resultado
LLM positivo no se llamará alfa sin costes, incertidumbre, trials reconciliados y una
replicación posterior.
