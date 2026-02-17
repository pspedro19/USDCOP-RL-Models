# =============================================================================
# USDCOP Trading System - Master Migration Runner (Windows PowerShell)
# =============================================================================
#
# This script applies all pending database migrations in order.
# It is IDEMPOTENT - safe to run multiple times.
#
# Usage:
#   .\run_all_migrations.ps1                   # Run against Docker container
#   .\run_all_migrations.ps1 -DryRun           # Show what would be applied
#   .\run_all_migrations.ps1 -Verbose          # Show detailed output
#
# Author: Trading Team
# Version: 1.0.0
# =============================================================================

param(
    [switch]$DryRun,
    [switch]$VerboseOutput,
    [string]$ContainerName = "usdcop-postgres-timescale",
    [string]$DbUser = "admin",
    [string]$DbName = "usdcop_trading"
)

$ErrorActionPreference = "Continue"

# =============================================================================
# CONFIGURATION
# =============================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function Write-ColorOutput {
    param([string]$Color, [string]$Message)
    $colors = @{
        "Red" = "Red"
        "Green" = "Green"
        "Yellow" = "Yellow"
        "Blue" = "Cyan"
    }
    Write-Host $Message -ForegroundColor $colors[$Color]
}

function Log-Info { param([string]$msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Log-Success { param([string]$msg) Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Log-Warning { param([string]$msg) Write-Host "[WARNING] $msg" -ForegroundColor Yellow }
function Log-Error { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Run-SQL {
    param([string]$sql)
    $result = docker exec $ContainerName psql -U $DbUser -d $DbName -t -c $sql 2>$null
    return $result
}

function Run-SQLFile {
    param([string]$filePath)
    $content = Get-Content $filePath -Raw
    $result = $content | docker exec -i $ContainerName psql -U $DbUser -d $DbName 2>&1
    return $result
}

function Is-MigrationApplied {
    param([string]$migrationName)
    $result = Run-SQL "SELECT COUNT(*) FROM _applied_migrations WHERE migration_name = '$migrationName';"
    $count = [int]($result.Trim())
    return $count -gt 0
}

function Mark-MigrationApplied {
    param([string]$migrationName, [string]$checksum)
    Run-SQL "INSERT INTO _applied_migrations (migration_name, checksum) VALUES ('$migrationName', '$checksum') ON CONFLICT (migration_name) DO NOTHING;" | Out-Null
}

function Get-FileChecksum {
    param([string]$filePath)
    $hash = Get-FileHash -Path $filePath -Algorithm MD5
    return $hash.Hash.Substring(0, 16)
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor White
Write-Host "   USDCOP Trading System - Database Migration Runner" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor White
Write-Host ""

# Check Docker container is running
$containerRunning = docker ps --format '{{.Names}}' | Select-String -Pattern "^$ContainerName$"
if (-not $containerRunning) {
    Log-Error "Docker container '$ContainerName' is not running!"
    Log-Info "Start it with: docker-compose up -d postgres-timescale"
    exit 1
}
Log-Info "Using Docker container: $ContainerName"

# Create migrations tracking table
Log-Info "Ensuring migrations tracking table exists..."
$createTableSQL = @"
CREATE TABLE IF NOT EXISTS _applied_migrations (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    checksum VARCHAR(64),
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    applied_by VARCHAR(100) DEFAULT CURRENT_USER
);
CREATE INDEX IF NOT EXISTS idx_migrations_name ON _applied_migrations(migration_name);
"@
Run-SQL $createTableSQL | Out-Null
Log-Success "Migrations tracking table ready"
Write-Host ""

# =============================================================================
# DEFINE MIGRATION ORDER
# =============================================================================

$migrations = @(
    # Phase 1: Core Extensions & Base Tables
    "init-scripts\00-init-extensions.sql",
    "init-scripts\01-essential-usdcop-init.sql",
    "init-scripts\02-macro-indicators-schema.sql",

    # Phase 2: Feature Store (CRITICAL for V7.1)
    "init-scripts\03-inference-features-views-v2.sql",

    # Phase 3: Model Registry & MLOps
    "init-scripts\05-model-registry.sql",
    "init-scripts\06-experiment-registry.sql",
    "init-scripts\10-multi-model-schema.sql",
    "init-scripts\11-paper-trading-tables.sql",
    "init-scripts\12-trades-metadata.sql",
    "init-scripts\15-forecasting-schema.sql",
    "init-scripts\20-signalbridge-schema.sql",

    # Phase 4: Incremental Migrations
    "database\migrations\020_feature_snapshot_improvements.sql",
    "database\migrations\021_drift_audit.sql",
    "database\migrations\022_experiment_registry.sql",
    "database\migrations\025_lineage_tables.sql",
    "database\migrations\026_v_macro_unified.sql",

    # Phase 5: V7.1 Event-Driven Architecture (CRITICAL)
    "database\migrations\033_event_triggers.sql",

    # Phase 6: Two-Vote Promotion System
    "database\migrations\034_promotion_proposals.sql",
    "database\migrations\035_approval_audit_log.sql",
    "database\migrations\036_model_registry_enhanced.sql",
    "database\migrations\037_experiment_contracts.sql"
)

# =============================================================================
# APPLY MIGRATIONS
# =============================================================================

$appliedCount = 0
$skippedCount = 0
$failedCount = 0

Write-Host "============================================================" -ForegroundColor White
Write-Host "   Applying Migrations" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor White
Write-Host ""

foreach ($migration in $migrations) {
    $migrationPath = Join-Path $ProjectRoot $migration
    $migrationName = Split-Path $migration -Leaf

    # Check if file exists
    if (-not (Test-Path $migrationPath)) {
        Log-Warning "Migration file not found: $migration"
        continue
    }

    # Check if already applied
    if (Is-MigrationApplied $migrationName) {
        if ($VerboseOutput) {
            Write-Host "  SKIP  $migrationName (already applied)" -ForegroundColor Yellow
        }
        $skippedCount++
        continue
    }

    # Get checksum
    $checksum = Get-FileChecksum $migrationPath

    if ($DryRun) {
        Write-Host "  WOULD APPLY  $migrationName" -ForegroundColor Cyan
        $appliedCount++
        continue
    }

    # Apply migration
    Write-Host -NoNewline "  Applying $migrationName... "

    try {
        $output = Run-SQLFile $migrationPath
        $hasError = $output -match "ERROR:"

        if (-not $hasError) {
            Mark-MigrationApplied $migrationName $checksum
            Write-Host "OK" -ForegroundColor Green
            $appliedCount++
        } else {
            Write-Host "FAILED" -ForegroundColor Red
            if ($VerboseOutput) {
                Write-Host $output -ForegroundColor Red
            }
            $failedCount++
        }
    } catch {
        Write-Host "FAILED" -ForegroundColor Red
        $failedCount++
    }
}

# =============================================================================
# SUMMARY
# =============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor White
Write-Host "   Migration Summary" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor White
Write-Host ""
Write-Host "  Applied:  $appliedCount" -ForegroundColor Green
Write-Host "  Skipped:  $skippedCount (already applied)" -ForegroundColor Yellow
Write-Host "  Failed:   $failedCount" -ForegroundColor Red
Write-Host ""

if ($DryRun) {
    Log-Warning "DRY RUN - No changes were made"
}

if ($failedCount -gt 0) {
    Log-Error "Some migrations failed. Run with -VerboseOutput for details."
}

if ($appliedCount -gt 0) {
    Log-Success "All pending migrations applied successfully!"
} else {
    Log-Info "Database is up to date - no new migrations to apply."
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor White
Write-Host "   Verification" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor White
Write-Host ""

# Quick verification
Log-Info "Checking critical tables..."

$tablesToCheck = @(
    "inference_features_5m",
    "event_dead_letter_queue",
    "circuit_breaker_state",
    "model_registry",
    "promotion_proposals"
)

foreach ($table in $tablesToCheck) {
    $exists = Run-SQL "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '$table';"
    $count = [int]($exists.Trim())
    if ($count -gt 0) {
        Write-Host "  [OK] $table" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $table" -ForegroundColor Red
    }
}

# Check triggers
Write-Host ""
Log-Info "Checking NOTIFY triggers..."
$triggerCount = Run-SQL "SELECT COUNT(*) FROM pg_trigger WHERE tgname LIKE '%notify%';"
$triggerCount = [int]($triggerCount.Trim())
Write-Host "  Found $triggerCount NOTIFY triggers" -ForegroundColor Cyan

Write-Host ""
Log-Success "Migration runner completed!"
Write-Host ""
