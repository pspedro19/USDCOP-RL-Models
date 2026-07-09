# scripts/cleanup_project.ps1
# Ejecutar desde la raíz del proyecto
# USO: .\scripts\cleanup_project.ps1 -WhatIf  # Para preview
#      .\scripts\cleanup_project.ps1          # Para ejecutar

param(
    [switch]$WhatIf = $false
)

$ErrorActionPreference = "Stop"
$totalRemoved = 0
$totalSize = 0

Write-Host "CLEANUP DEL PROYECTO USDCOP-RL-Models" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Eliminar directorios tmpclaude-*
Write-Host "`nBuscando directorios tmpclaude-*..." -ForegroundColor Yellow
$tmpDirs = Get-ChildItem -Path . -Directory -Recurse -Filter "tmpclaude-*" -ErrorAction SilentlyContinue
if ($tmpDirs) {
    $count = $tmpDirs.Count
    $size = ($tmpDirs | Get-ChildItem -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "  Encontrados: $count directorios (~$([math]::Round($size, 2)) MB)" -ForegroundColor Red

    if (-not $WhatIf) {
        $tmpDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Eliminados" -ForegroundColor Green
    } else {
        Write-Host "  [WhatIf] Se eliminarían $count directorios" -ForegroundColor Gray
    }
    $totalRemoved += $count
    $totalSize += $size
}

# 2. Eliminar __pycache__
Write-Host "`nBuscando __pycache__..." -ForegroundColor Yellow
$pycacheDirs = Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue
if ($pycacheDirs) {
    $count = $pycacheDirs.Count
    Write-Host "  Encontrados: $count directorios" -ForegroundColor Red

    if (-not $WhatIf) {
        $pycacheDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Eliminados" -ForegroundColor Green
    }
    $totalRemoved += $count
}

# 3. Eliminar .pyc huérfanos
Write-Host "`nBuscando archivos .pyc..." -ForegroundColor Yellow
$pycFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue
if ($pycFiles) {
    $count = $pycFiles.Count
    Write-Host "  Encontrados: $count archivos" -ForegroundColor Red

    if (-not $WhatIf) {
        $pycFiles | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host "  Eliminados" -ForegroundColor Green
    }
}

# 4. Eliminar .pytest_cache
Write-Host "`nBuscando .pytest_cache..." -ForegroundColor Yellow
$pytestDirs = Get-ChildItem -Path . -Directory -Recurse -Filter ".pytest_cache" -ErrorAction SilentlyContinue
if ($pytestDirs) {
    $count = $pytestDirs.Count
    Write-Host "  Encontrados: $count directorios" -ForegroundColor Red

    if (-not $WhatIf) {
        $pytestDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Eliminados" -ForegroundColor Green
    }
    $totalRemoved += $count
}

# 5. Eliminar node_modules/.cache
Write-Host "`nBuscando node_modules/.cache..." -ForegroundColor Yellow
$nmCache = Get-ChildItem -Path . -Directory -Recurse -Filter ".cache" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -like "*node_modules*" }
if ($nmCache) {
    $count = $nmCache.Count
    Write-Host "  Encontrados: $count directorios" -ForegroundColor Red

    if (-not $WhatIf) {
        $nmCache | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Eliminados" -ForegroundColor Green
    }
}

# 6. Crear directorio archive/ si no existe
Write-Host "`nVerificando directorio archive/..." -ForegroundColor Yellow
if (-not (Test-Path "archive")) {
    if (-not $WhatIf) {
        New-Item -ItemType Directory -Path "archive" | Out-Null
        Write-Host "  Creado archive/" -ForegroundColor Green
    } else {
        Write-Host "  [WhatIf] Se crearía archive/" -ForegroundColor Gray
    }
}

# 7. Mover docs obsoletos a archive/
$obsoleteDocs = @(
    "CLEANUP_COMPLETE_SUMMARY.md",
    "NEXT_STEPS_COMPLETE.md",
    "REPLAY_SYSTEM_*.md"
)

Write-Host "`nArchivando documentos obsoletos..." -ForegroundColor Yellow
foreach ($pattern in $obsoleteDocs) {
    $files = Get-ChildItem -Path . -Filter $pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        if (-not $WhatIf) {
            Move-Item $file.FullName "archive/" -Force -ErrorAction SilentlyContinue
            Write-Host "  Archivado: $($file.Name)" -ForegroundColor Green
        } else {
            Write-Host "  [WhatIf] Se archivaría: $($file.Name)" -ForegroundColor Gray
        }
    }
}

# Resumen
Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "RESUMEN" -ForegroundColor Cyan
Write-Host "  Directorios procesados: $totalRemoved" -ForegroundColor White
Write-Host "  Espacio estimado liberado: ~$([math]::Round($totalSize, 2)) MB" -ForegroundColor White

if ($WhatIf) {
    Write-Host "`nEjecutado en modo WhatIf - nada fue eliminado" -ForegroundColor Yellow
    Write-Host "   Ejecutar sin -WhatIf para aplicar cambios" -ForegroundColor Yellow
}

Write-Host "`nCleanup completado" -ForegroundColor Green
