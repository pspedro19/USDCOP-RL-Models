# Cierra EXP-TESIS-RL-02 de una vez, cuando los dos brazos LLM hayan terminado.
#
# Orden obligado: primero se recuperan las barras que el proveedor no respondio (si no, quedan
# selladas como huecos y sus sesiones se pierden), y solo despues se liquida. Liquidar antes
# produciria un informe correcto sobre una muestra mutilada.
#
# Uso:
#   .\scripts\tools\finish_thesis_v2.ps1                # cadena completa
#   .\scripts\tools\finish_thesis_v2.ps1 -SkipRepair    # si ya se recuperaron las caidas
param(
    [switch]$SkipRepair,
    [string]$Block = 'selection'
)
$ErrorActionPreference = 'Stop'
$out = 'outputs/thesis-repair'
$ppo = "$out/ppo_v2_recipe_flat"

$arms = @(
    @{ name = 'deepseek';     ledger = 'data/thesis/llm/decisions_deepseek_selection_diagnostic.jsonl'; provider = 'deepseek';     model = 'deepseek-chat' },
    @{ name = 'azure';        ledger = 'data/thesis/llm/decisions_azure_selection_diagnostic.jsonl';    provider = 'azure_openai'; model = 'gpt-4o-mini' }
)

if (-not $SkipRepair) {
    foreach ($a in $arms) {
        Write-Host "== recuperando barras sin respuesta: $($a.name)"
        python scripts/tools/quarantine_unavailable_decisions.py --ledger $a.ledger
        if ($LASTEXITCODE -eq 3) { throw "El ledger de $($a.name) sigue activo. Espera a que termine el brazo." }
        $env:USDCOP_AZURE_OPENAI_DEPLOYMENT = 'gpt-4o-mini'
        python scripts/analysis/run_thesis_llm.py --input-jsonl "$out/llm_selection_contexts_v2.jsonl" `
            --portable data/thesis/research_data_portable_v2.pkl --allow-retrospective --load-dotenv `
            --provider $a.provider --model-id $a.model --ledger $a.ledger --resume --execute
    }
}

foreach ($a in $arms) {
    Write-Host "== liquidando $($a.name)"
    python scripts/analysis/settle_thesis_llm.py --ledger $a.ledger --block $Block `
        --dataset-version v2 --output "$out/settlement_$($a.name)_${Block}_v2.json"
    Write-Host "== hibrido $($a.name)"
    python scripts/analysis/thesis_hybrid.py --ppo-weights "$out/ppo_v2_weights_$Block.json" `
        --ledger $a.ledger --block $Block --dataset-version v2 `
        --output "$out/hybrid_$($a.name)_${Block}_v2.json"
}

Write-Host '== estadistica con todos los brazos'
$env:THESIS_PPO_OUT = $ppo
$env:THESIS_RESULTS_OUT = "$out/results_v2"
python scripts/analysis/thesis_statistics.py --block $Block --dataset-version v2 `
    --extra-series "deepseek=$out/settlement_deepseek_${Block}_v2.json" `
    --extra-series "azure=$out/settlement_azure_${Block}_v2.json" `
    --extra-series "hibrido_ds=$out/hybrid_deepseek_${Block}_v2.json" `
    --extra-series "hibrido_azure=$out/hybrid_azure_${Block}_v2.json"

Write-Host '== figuras comparativas'
python scripts/presentation/generar_figuras_comparativas.py `
    --statistics "$out/results_v2/statistics_$Block.json" `
    --settlement "deepseek=$out/settlement_deepseek_${Block}_v2.json" `
    --settlement "azure=$out/settlement_azure_${Block}_v2.json" `
    --settlement "hibrido_ds=$out/hybrid_deepseek_${Block}_v2.json" `
    --settlement "hibrido_azure=$out/hybrid_azure_${Block}_v2.json" `
    --ppo-dir $ppo --output-dir "$out/results_v2/figuras"

Write-Host '== estado e2e'
python scripts/diagnostics/audit_thesis_e2e_status.py --output "$out/e2e_status_final.json"
