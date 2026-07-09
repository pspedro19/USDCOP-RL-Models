# Disable Investor Mode - Return to Real Backtest
# Run this after investor meetings to return to normal operation

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  DISABLING INVESTOR MODE              " -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

# Update .env file
$envPath = Join-Path $PSScriptRoot ".." ".env"
$content = Get-Content $envPath -Raw
$content = $content -replace "INVESTOR_MODE=true", "INVESTOR_MODE=false"
Set-Content $envPath $content

Write-Host "[OK] Updated .env: INVESTOR_MODE=false" -ForegroundColor Green

# Restart backtest-api with new environment
Write-Host ""
Write-Host "Restarting backtest-api service..." -ForegroundColor Yellow
Set-Location (Join-Path $PSScriptRoot "..")
docker-compose up -d backtest-api

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  NORMAL MODE RESTORED                 " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backtest will now use real model inference." -ForegroundColor Cyan
Write-Host ""
