# Serie v2 con la receta que la compuerta de sanidad selecciono (`flat_init_no_turn`).
# Se separa del brazo de control (`ppo_v2_diagnostic_full`, que corre Codex sin el sesgo
# inicial) para que el contraste sea de UNA variable sobre el mismo dato y las mismas semillas.
$ErrorActionPreference = 'Continue'
$out = 'outputs/thesis-repair/ppo_v2_recipe_flat'
New-Item -ItemType Directory -Force -Path $out | Out-Null
& python scripts/analysis/thesis_train_ppo.py --all --dataset-version v2 `
    --diagnostic-retrospective --portable-path data/thesis/research_data_portable_v2.pkl `
    --output-dir $out *>&1 | Out-File -Append -Encoding utf8 "$out/run.log"
"=== PPO_V2_RECIPE_DONE exit=$LASTEXITCODE $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 "$out/run.log"
