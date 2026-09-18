# Ejecución reproducible del brazo LLM de la tesis

Este runbook describe el orden operativo del brazo LLM. Las claves se mantienen
fuera del repositorio y nunca se escriben en el ledger.

## 1. Preflight sin red ni trials

Antes del preflight LLM, el operador debe cerrar la identidad macro. Para un
el SSOT actual es el instrumento Investing 942611; el proxy FRED DTWEXBGS
solo se conserva como diagnóstico. La comprobación es:

```powershell
python scripts/diagnostics/verify_macro_declared_identity.py `
  --output outputs/thesis-repair/macro_identity_research_v2_latest.json
```

El resultado solo puede usarse para desbloquear el siguiente gate si reporta
`dxy.status = COINCIDE`, `instrument_id = 942611`, una fracción de coincidencia
superior al 99 %, hash del payload y `all_declared_identities_honoured = true`.
Un proxy FRED o una correlación alta nunca satisface esta condición.

```powershell
python scripts/validation/check_thesis_llm_readiness.py --provider deepseek
python scripts/validation/check_thesis_llm_readiness.py --provider azure_openai
python scripts/diagnostics/thesis_plan_status.py --output outputs/thesis-repair/plan_status_before_llm.json
```

Para una ejecución local donde el operador guarda las variables en `.env`, usar
explícitamente:

```powershell
python scripts/validation/check_thesis_llm_readiness.py --provider deepseek --load-dotenv
python scripts/validation/check_thesis_llm_readiness.py --provider azure_openai --load-dotenv
```

Si el `.env` del workspace no es accesible por permisos, el operador puede
proporcionar otro archivo protegido fuera del repositorio:

```powershell
python scripts/validation/check_thesis_llm_readiness.py `
  --provider deepseek --load-dotenv --dotenv-path C:\ruta\segura\thesis.env
```

El readiness solo informa presencia/ausencia. No imprime valores y no llama a un
proveedor.

La identidad macro y la frescura son compuertas distintas: la primera puede desbloquear
la reconstrucción histórica v2; el carril confirmatorio/forward además exige que **cada**
serie declarada esté dentro de `max_staleness_business_days`. Una fila reciente de IBR o
DGS2 no oculta un DXY atrasado. Si Investing no publica DXY hasta la última sesión, el
forward queda explícitamente bloqueado y no se reutiliza el último valor como si fuera
actual.

## 2. Generar contextos

Los contextos confirmatorios solo se generan desde el portable v2 que tenga la
identidad vigente. Para una prueba retrospectiva, usar un directorio bajo
`outputs/thesis-repair/` y mantener `--allow-retrospective`:

```powershell
python scripts/analysis/export_thesis_llm_contexts.py `
  --block selection `
  --portable outputs/thesis-repair/research_data_portable_diagnostic_v2.pkl `
  --output outputs/thesis-repair/llm_selection_contexts_v2_diagnostic.jsonl `
  --allow-retrospective
```

Nunca copiar esos contextos a `data/thesis/llm/selection_contexts.jsonl` ni
usarlos como confirmación.

Para ejecutar las dos variantes diagnósticas ya exportadas, sin introducir claves en el
repositorio, usar el lanzador [run_thesis_llm_dual.ps1](../../scripts/tools/run_thesis_llm_dual.ps1).
Exige fijar explícitamente el deployment de Azure y conserva ledgers separados. La opción
`-Execute` es deliberadamente necesaria para habilitar llamadas de red:

```powershell
.\scripts\tools\run_thesis_llm_dual.ps1 `
  -DotenvPath C:\ruta\segura\thesis.env `
  -AzureModelId NOMBRE_EXACTO_DEL_DEPLOYMENT `
  -Limit 10

.\scripts\tools\run_thesis_llm_dual.ps1 `
  -DotenvPath C:\ruta\segura\thesis.env `
  -AzureModelId NOMBRE_EXACTO_DEL_DEPLOYMENT `
  -Execute
```

El primer comando valida diez contextos sin red; el segundo procesa los 13.334 contextos
retrospectivos de selección. Ambos resultados son diagnósticos y no sustituyen el juez
forward. Nunca se elige retrospectivamente el proveedor por PnL.

## 3. Ejecutar un proveedor congelado

Antes de ejecutar, registrar en el preregistro el proveedor, modelo/deployment,
versión del prompt y parámetros de muestreo. DeepSeek es el proveedor principal;
Azure es robustez. No se selecciona retrospectivamente el que tenga mejor PnL.

```powershell
python scripts/analysis/run_thesis_llm.py `
  --input-jsonl data/thesis/llm/selection_contexts.jsonl `
  --provider deepseek `
  --model-id deepseek-chat `
  --ledger data/thesis/llm/decisions_deepseek.jsonl `
  --load-dotenv --dotenv-path C:\ruta\segura\thesis.env `
  --execute
```

Para Azure:

```powershell
python scripts/analysis/run_thesis_llm.py `
  --input-jsonl data/thesis/llm/selection_contexts.jsonl `
  --provider azure_openai `
  --model-id <deployment-fijado-en-preregistro> `
  --ledger data/thesis/llm/decisions_azure.jsonl `
  --load-dotenv `
  --execute
```

El runner rechaza hashes de dataset incorrectos, contextos retrospectivos sin
`--allow-retrospective`, duplicados y parámetros que no coincidan con el contrato.

## 4. Validar y liquidar

```powershell
python scripts/validation/validate_thesis_llm_ledger.py `
  --input data/thesis/llm/decisions_deepseek.jsonl

python scripts/analysis/settle_thesis_llm.py `
  --ledger data/thesis/llm/decisions_deepseek.jsonl `
  --strict-ledger
```

Una sesión con barras no selladas es `NA`, nunca retorno cero. El ledger debe
conservar hashes de prompt y respuesta, proveedor, modelo, parámetros, latencia,
acción, exposición previa y dataset.

## 5. Figuras

Las curvas de capital, drawdown y Sharpe solo se generan desde un ledger validado
y liquidado. Si falta el ledger, el generador debe producir un error y no una
figura con ceros o datos simulados.

## 6. Regla de bloqueo

No ejecutar el carril confirmatorio mientras `thesis_plan_status.py` reporte:

```text
ready_for_confirmatory_llm = false
```

En particular, la identidad macro (incluido DXY/Investing 942611), el preregistro firmado y la
identidad del portable v2 deben estar cerrados antes de abrir el experimento. Esta coincidencia
es de reproducibilidad con la fuente declarada; no equivale a validación independiente contra
ICE ni a una licencia de redistribución.
