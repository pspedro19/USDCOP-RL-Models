<#
.SYNOPSIS
    Emergency Rollback Script for USDCOP Trading Platform

.DESCRIPTION
    Performs immediate rollback to the previous stable version.
    Skips health checks for emergency scenarios.

.PARAMETER Force
    Skip all confirmations and health checks

.EXAMPLE
    .\rollback.ps1
    Performs rollback with confirmation

.EXAMPLE
    .\rollback.ps1 -Force
    Performs immediate rollback without confirmation

.NOTES
    Contract: CTR-DEPLOY-001
    Author: USDCOP Trading Platform
#>

param(
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$NginxConfigPath = Join-Path $ProjectRoot "nginx\conf.d\active.conf"
$LogPath = Join-Path $ProjectRoot "logs\deployment"

# Ensure log directory exists
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
}

$LogFile = Join-Path $LogPath "rollback_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    Write-Host $LogEntry -ForegroundColor $(switch ($Level) { "ERROR" { "Red" } "WARN" { "Yellow" } default { "White" } })
    Add-Content -Path $LogFile -Value $LogEntry
}

function Get-CurrentBackend {
    if (Test-Path $NginxConfigPath) {
        $Content = Get-Content $NginxConfigPath -Raw
        if ($Content -match 'default "inference_(\w+)"') {
            return $Matches[1]
        }
    }
    return "blue"
}

function Get-RollbackTarget {
    $Current = Get-CurrentBackend
    return if ($Current -eq "blue") { "green" } else { "blue" }
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

Write-Log "=========================================="
Write-Log "EMERGENCY ROLLBACK INITIATED" "WARN"
Write-Log "=========================================="

$CurrentBackend = Get-CurrentBackend
$RollbackTarget = Get-RollbackTarget

Write-Log "Current active backend: $CurrentBackend"
Write-Log "Rollback target: $RollbackTarget"

if (-not $Force) {
    Write-Host ""
    Write-Host "WARNING: This will immediately switch traffic to $RollbackTarget" -ForegroundColor Yellow
    Write-Host ""
    $Confirm = Read-Host "Type 'ROLLBACK' to confirm"

    if ($Confirm -ne "ROLLBACK") {
        Write-Log "Rollback cancelled by user" "WARN"
        exit 0
    }
}

Write-Log "Executing rollback to $RollbackTarget..."

try {
    # Call the blue-green script with health check disabled for speed
    $DeployScript = Join-Path $PSScriptRoot "deploy_blue_green.ps1"

    & $DeployScript -Target $RollbackTarget -HealthCheck $false

    Write-Log "=========================================="
    Write-Log "ROLLBACK COMPLETED SUCCESSFULLY"
    Write-Log "Traffic now routing to: $RollbackTarget"
    Write-Log "=========================================="

    exit 0
}
catch {
    Write-Log "ROLLBACK FAILED: $($_.Exception.Message)" "ERROR"
    Write-Log "MANUAL INTERVENTION REQUIRED" "ERROR"
    exit 1
}
