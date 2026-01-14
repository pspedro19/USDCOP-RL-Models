<#
.SYNOPSIS
    Blue-Green Deployment Script for USDCOP Trading Platform

.DESCRIPTION
    This script manages blue-green deployments by switching traffic
    between blue and green environments with zero downtime.

.PARAMETER Target
    The target environment to switch to (blue or green)

.PARAMETER HealthCheck
    Perform health check before switching (default: true)

.PARAMETER Timeout
    Health check timeout in seconds (default: 30)

.EXAMPLE
    .\deploy_blue_green.ps1 -Target green
    Switches traffic to green environment

.EXAMPLE
    .\deploy_blue_green.ps1 -Target blue -HealthCheck $false
    Switches to blue without health check (emergency rollback)

.NOTES
    Contract: CTR-DEPLOY-001
    Author: USDCOP Trading Platform
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("blue", "green")]
    [string]$Target,

    [Parameter(Mandatory=$false)]
    [bool]$HealthCheck = $true,

    [Parameter(Mandatory=$false)]
    [int]$Timeout = 30
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

$LogFile = Join-Path $LogPath "deployment_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    Write-Host $LogEntry
    Add-Content -Path $LogFile -Value $LogEntry
}

function Test-ServiceHealth {
    param([string]$Color, [int]$TimeoutSeconds)

    $Port = if ($Color -eq "blue") { 8091 } else { 8092 }
    $Url = "http://localhost:$Port/v1/health"

    Write-Log "Checking health of $Color environment at $Url"

    $StartTime = Get-Date
    $EndTime = $StartTime.AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $EndTime) {
        try {
            $Response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 5
            if ($Response.status -eq "healthy" -or $Response.status -eq "ok") {
                Write-Log "$Color environment is healthy"
                return $true
            }
        }
        catch {
            Write-Log "Health check failed, retrying... ($($_.Exception.Message))" "WARN"
        }
        Start-Sleep -Seconds 2
    }

    Write-Log "$Color environment failed health check after $TimeoutSeconds seconds" "ERROR"
    return $false
}

function Get-CurrentBackend {
    if (Test-Path $NginxConfigPath) {
        $Content = Get-Content $NginxConfigPath -Raw
        if ($Content -match 'default "inference_(\w+)"') {
            return $Matches[1]
        }
    }
    return "blue"  # Default
}

function Set-ActiveBackend {
    param([string]$Color)

    $Timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    $CurrentUser = $env:USERNAME

    $NewConfig = @"
# =============================================================================
# ACTIVE BACKEND CONFIGURATION
# =============================================================================
#
# This file controls which backend receives production traffic.
# Modified by deploy_blue_green.ps1 script.
#
# DO NOT EDIT MANUALLY - Use deployment scripts instead.
#
# Contract: CTR-DEPLOY-001
# =============================================================================

# Active backend: blue or green
# This variable is used in the main nginx.conf to route traffic

map `$host `$active_backend {
    default "inference_$Color";
}

# Deployment metadata (for audit trail)
# Last switched: $Timestamp
# Switched by: $CurrentUser
# Previous: $(Get-CurrentBackend)
"@

    Set-Content -Path $NginxConfigPath -Value $NewConfig
    Write-Log "Updated nginx configuration to route to $Color"
}

function Restart-NginxLoadBalancer {
    Write-Log "Reloading nginx configuration..."

    try {
        # For Docker deployment
        docker exec usdcop-nginx-lb nginx -s reload 2>$null
        Write-Log "Nginx reloaded successfully (Docker)"
        return $true
    }
    catch {
        Write-Log "Docker nginx reload failed, trying docker-compose..." "WARN"
    }

    try {
        # Alternative: restart via docker-compose
        Push-Location $ProjectRoot
        docker-compose -f docker-compose.blue-green.yml restart nginx-lb
        Pop-Location
        Write-Log "Nginx restarted via docker-compose"
        return $true
    }
    catch {
        Write-Log "Failed to reload nginx: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

Write-Log "=========================================="
Write-Log "Blue-Green Deployment Starting"
Write-Log "Target: $Target"
Write-Log "Health Check: $HealthCheck"
Write-Log "=========================================="

$CurrentBackend = Get-CurrentBackend
Write-Log "Current active backend: $CurrentBackend"

if ($CurrentBackend -eq $Target) {
    Write-Log "Traffic is already routing to $Target. No action needed."
    exit 0
}

# Health check if enabled
if ($HealthCheck) {
    Write-Log "Performing health check on $Target environment..."

    if (-not (Test-ServiceHealth -Color $Target -TimeoutSeconds $Timeout)) {
        Write-Log "DEPLOYMENT ABORTED: $Target environment is not healthy" "ERROR"
        exit 1
    }
}

# Switch traffic
Write-Log "Switching traffic from $CurrentBackend to $Target..."

try {
    Set-ActiveBackend -Color $Target

    if (Restart-NginxLoadBalancer) {
        Write-Log "=========================================="
        Write-Log "DEPLOYMENT SUCCESSFUL"
        Write-Log "Traffic now routing to: $Target"
        Write-Log "Previous backend: $CurrentBackend"
        Write-Log "=========================================="

        # Verify the switch
        Start-Sleep -Seconds 2
        try {
            $StatusResponse = Invoke-RestMethod -Uri "http://localhost:8090/lb-status" -Method Get
            Write-Log "Load balancer status: $($StatusResponse | ConvertTo-Json -Compress)"
        }
        catch {
            Write-Log "Could not verify load balancer status" "WARN"
        }

        exit 0
    }
    else {
        throw "Nginx reload failed"
    }
}
catch {
    Write-Log "DEPLOYMENT FAILED: $($_.Exception.Message)" "ERROR"
    Write-Log "Attempting rollback to $CurrentBackend..." "WARN"

    Set-ActiveBackend -Color $CurrentBackend
    Restart-NginxLoadBalancer

    Write-Log "Rollback completed. Traffic still routing to $CurrentBackend"
    exit 1
}
