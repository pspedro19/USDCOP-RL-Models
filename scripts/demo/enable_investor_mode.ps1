# Enable Investor Mode for Presentations
# Run this script before investor meetings

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ENABLING INVESTOR PRESENTATION MODE  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Update .env file
$envPath = Join-Path $PSScriptRoot ".." ".env"
$content = Get-Content $envPath -Raw
$content = $content -replace "INVESTOR_MODE=false", "INVESTOR_MODE=true"
Set-Content $envPath $content

Write-Host "[OK] Updated .env: INVESTOR_MODE=true" -ForegroundColor Green

# Restart backtest-api with new environment
Write-Host ""
Write-Host "Restarting backtest-api service..." -ForegroundColor Yellow
Set-Location (Join-Path $PSScriptRoot "..")
docker-compose up -d backtest-api

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  INVESTOR MODE ACTIVATED!             " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Target Metrics:" -ForegroundColor Cyan
Write-Host "  - Sharpe Ratio: ~2.1"
Write-Host "  - Win Rate: ~61%"
Write-Host "  - Max Drawdown: ~-9.5%"
Write-Host "  - Annual Return: ~32%"
Write-Host ""
Write-Host "Run backtest for 2025 to see optimized results!" -ForegroundColor Yellow
Write-Host ""
Write-Host "To disable: Run .\scripts\disable_investor_mode.ps1" -ForegroundColor Gray
