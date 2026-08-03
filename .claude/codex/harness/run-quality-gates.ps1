param(
    [ValidateSet('static', 'full')]
    [string]$Mode = 'static'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$dashboardRoot = Join-Path $repoRoot 'usdcop-trading-dashboard'
$evidenceRoot = Join-Path $PSScriptRoot '..\evidence'
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$runRoot = Join-Path $evidenceRoot $runId
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$results = [System.Collections.Generic.List[object]]::new()

function Invoke-Gate {
    param([string]$Name, [string]$WorkingDirectory, [scriptblock]$Command)
    $logPath = Join-Path $runRoot "$Name.log"
    $started = Get-Date
    $exitCode = 0
    try {
        Push-Location $WorkingDirectory
        & $Command *>&1 | Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne $null) { $exitCode = $LASTEXITCODE }
    } catch {
        $_ | Out-String | Add-Content -LiteralPath $logPath
        $exitCode = 1
    } finally {
        Pop-Location
    }
    $results.Add([pscustomobject]@{
        gate = $Name
        exit_code = $exitCode
        duration_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 2)
        log = [IO.Path]::GetRelativePath($repoRoot, $logPath)
    })
}

Invoke-Gate 'spec-contracts' $repoRoot {
    python -m pytest tests/regression/test_knowledge_frontmatter.py `
        tests/regression/test_knowledge_inventory.py `
        tests/regression/test_contract_mirrors.py `
        tests/regression/test_dev_auth_bypass_guard.py -q
}
Invoke-Gate 'codex-assurance-contracts' $repoRoot {
    python -m pytest .claude/codex/harness/tests -q
}
Invoke-Gate 'python-lint' $repoRoot { python -m ruff check src services tests }
Invoke-Gate 'frontend-lint' $dashboardRoot { npm run lint }
Invoke-Gate 'frontend-unit' $dashboardRoot { npm run test:run }
Invoke-Gate 'frontend-build' $dashboardRoot { npm run build }
Invoke-Gate 'rbac-gate' $dashboardRoot { npm run qa:gate }

if ($Mode -eq 'full') {
    Invoke-Gate 'codex-visual-evidence' $dashboardRoot {
        npx playwright test --config ..\.claude\codex\harness\playwright.codex.config.ts
    }
    python (Join-Path $PSScriptRoot 'normalize_evidence_frontmatter.py') `
        (Join-Path $PSScriptRoot '..\evidence\playwright')
    Invoke-Gate 'frontend-e2e' $dashboardRoot {
        $env:PLAYWRIGHT_HTML_REPORT = (Join-Path $runRoot 'playwright-report')
        npm run test:e2e
    }
    Invoke-Gate 'python-security-checklist' $repoRoot {
        python scripts/validation/pentest_checklist.py
    }
}

$manifest = [pscustomobject]@{
    schema_version = 1
    run_id = $runId
    mode = $Mode
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    commit = (git -C $repoRoot rev-parse HEAD)
    dirty = [bool](git -C $repoRoot status --porcelain)
    results = $results
    passed = -not [bool]($results | Where-Object { $_.exit_code -ne 0 })
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'manifest.json')

if (-not $manifest.passed) { exit 1 }
