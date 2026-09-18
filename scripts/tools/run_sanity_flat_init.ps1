# Corre la compuerta de sanidad completa con la receta que abrio S1 (`flat_init`), en CPU.
# S1 se re-corre a proposito: sus cinco semillas se midieron con la rama fresca en GPU y la
# reanudada en CPU, asi que la compuerta no era homogenea. Aqui las cuatro fixtures usan
# exactamente la misma receta y el mismo dispositivo.
$ErrorActionPreference = 'Continue'
$out = 'outputs/thesis-repair/sanity_flat_init'
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($f in @('S2','S3','S4','S1')) {
  "=== INICIO $f $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 "$out/run.log"
  & python scripts/analysis/thesis_ppo_sanity.py --fixture $f --probe flat_init --output "$out/${f}_flat_init.json" *>&1 |
      Out-File -Append -Encoding utf8 "$out/run.log"
  "=== FIXTURE_DONE $f exit=$LASTEXITCODE $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 "$out/run.log"
}
"=== SANITY_ALL_DONE $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 "$out/run.log"
