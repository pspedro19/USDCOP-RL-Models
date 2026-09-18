[CmdletBinding()]
param(
    [string]$DotenvPath,
    [string]$AzureModelId,
    [switch]$Execute,
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$python = Join-Path $repo 'python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = 'python'
}

$contexts = Join-Path $repo 'outputs/thesis-repair/llm_selection_contexts_v2.jsonl'
$portable = Join-Path $repo 'data/thesis/research_data_portable_v2.pkl'
$deepseekLedger = Join-Path $repo 'data/thesis/llm/decisions_deepseek_selection_diagnostic.jsonl'
$azureLedger = Join-Path $repo 'data/thesis/llm/decisions_azure_selection_diagnostic.jsonl'

if (-not (Test-Path -LiteralPath $contexts)) { throw "No existe el contexto sellado: $contexts" }
if (-not (Test-Path -LiteralPath $portable)) { throw "No existe el portable v2: $portable" }
if ([string]::IsNullOrWhiteSpace($AzureModelId)) { throw 'Debes fijar -AzureModelId con el deployment exacto de Azure.' }

$common = @(
    'scripts/analysis/run_thesis_llm.py',
    '--input-jsonl', $contexts,
    '--portable', $portable,
    '--allow-retrospective',
    '--load-dotenv',
    '--provider', 'deepseek',
    '--model-id', 'deepseek-chat',
    '--ledger', $deepseekLedger
)
if (-not [string]::IsNullOrWhiteSpace($DotenvPath)) { $common += @('--dotenv-path', $DotenvPath) }
if ($Limit -gt 0) { $common += @('--limit', $Limit) }
if ($Execute) { $common += '--execute' }

Write-Host 'DeepSeek: ejecución explícita sobre contextos retrospectivos (no confirmatoria).'
& $python @common
if ($LASTEXITCODE -ne 0) { throw "DeepSeek terminó con código $LASTEXITCODE" }

$azure = @(
    'scripts/analysis/run_thesis_llm.py',
    '--input-jsonl', $contexts,
    '--portable', $portable,
    '--allow-retrospective',
    '--load-dotenv',
    '--provider', 'azure_openai',
    '--model-id', $AzureModelId,
    '--ledger', $azureLedger
)
if (-not [string]::IsNullOrWhiteSpace($DotenvPath)) { $azure += @('--dotenv-path', $DotenvPath) }
if ($Limit -gt 0) { $azure += @('--limit', $Limit) }
if ($Execute) { $azure += '--execute' }

Write-Host 'Azure OpenAI: ejecución explícita sobre contextos retrospectivos (no confirmatoria).'
& $python @azure
if ($LASTEXITCODE -ne 0) { throw "Azure OpenAI terminó con código $LASTEXITCODE" }

Write-Host 'Listo. Valida ambos ledgers antes de liquidar; no selecciones proveedor por PnL.'
