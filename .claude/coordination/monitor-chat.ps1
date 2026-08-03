[CmdletBinding()]
param(
    [int]$RefreshSeconds = 2,
    [int]$MessageCount = 24
)

$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $root '..\..')
$inboxCodex = Join-Path $repoRoot '.claude\coordination\INBOX-CODEX.md'
$inboxClaude = Join-Path $repoRoot '.claude\coordination\INBOX-CLAUDE.md'
$statusClaude = Join-Path $repoRoot '.claude\coordination\CLAUDE-STATUS.md'
$monitorLog = Join-Path $repoRoot '.claude\coordination\monitor\CLAUDE-MESSAGES.log'

function Get-ChatMessages {
    param([string]$Path, [ValidateSet('CLAUDE','CODEX')][string]$Speaker)
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    $lines = Get-Content -LiteralPath $Path -Encoding utf8
    $result = New-Object System.Collections.Generic.List[object]
    foreach ($line in $lines) {
        if ($line -match '^\- \[(?<id>(?:CLD|CXD)-\d+)\].*') {
            $title = $line.Substring(2).Trim()
            $result.Add([pscustomobject]@{
                Id = $Matches.id
                Speaker = $Speaker
                Text = $title
                SortKey = [int]($Matches.id -replace '^(CLD|CXD)-','')
            })
        }
    }
    return $result
}

function Get-Value([string]$Path, [string]$Pattern, [string]$Fallback) {
    if (-not (Test-Path -LiteralPath $Path)) { return $Fallback }
    $hit = Select-String -LiteralPath $Path -Pattern $Pattern | Select-Object -Last 1
    if ($hit) { return ($hit.Line -replace '^.*?:\s*','').Trim() }
    return $Fallback
}

function Draw {
    Clear-Host
    $now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $state = Get-Value $statusClaude '^estado:' 'UNKNOWN'
    $heartbeat = Get-Value $statusClaude '^timestamp:' 'UNKNOWN'
    $monitor = if (Test-Path -LiteralPath $monitorLog) {
        (Get-Content -LiteralPath $monitorLog -Encoding utf8 -Tail 1)
    } else { 'monitor log unavailable' }

    Write-Host '+----------------------------------------------------------------------+' -ForegroundColor Cyan
    Write-Host '|              CODEX ↔ CLAUDE  |  LIVE COORDINATION CHAT              |' -ForegroundColor Cyan
    Write-Host '+----------------------------------------------------------------------+' -ForegroundColor Cyan
    Write-Host ("| {0,-14} Claude: {1,-46} |" -f 'LOCAL '+$now.Substring(11), $state.Substring(0,[Math]::Min(46,$state.Length))) -ForegroundColor White
    Write-Host ("| {0,-14} Heartbeat: {1,-42} |" -f 'Monitor', $heartbeat.Substring(0,[Math]::Min(42,$heartbeat.Length))) -ForegroundColor DarkGray
    Write-Host '+----------------------------------------------------------------------+' -ForegroundColor Cyan

    $messages = @(
        (Get-ChatMessages $inboxClaude 'CODEX')
        (Get-ChatMessages $inboxCodex 'CLAUDE')
    ) | Sort-Object SortKey -Descending | Select-Object -First $MessageCount | Sort-Object SortKey

    foreach ($m in $messages) {
        $color = if ($m.Speaker -eq 'CODEX') { 'Green' } else { 'Yellow' }
        $label = if ($m.Speaker -eq 'CODEX') { 'CODEX ->' } else { 'CLAUDE <-' }
        $text = $m.Text -replace '[\r\n]+',' '
        if ($text.Length -gt 57) { $text = $text.Substring(0,57) + '...' }
        Write-Host ("| {0,-9} {1,-8} {2,-49} |" -f $label,$m.Id,$text) -ForegroundColor $color
    }
    Write-Host '+----------------------------------------------------------------------+' -ForegroundColor Cyan
    $monitorText = $monitor -replace '^\[[^\]]+\]\s*',''
    if ($monitorText.Length -gt 65) { $monitorText = $monitorText.Substring(0,65) + '...' }
    Write-Host ("| MONITOR  {0,-58} |" -f $monitorText) -ForegroundColor DarkCyan
    Write-Host '| [R] refresh  [P] pause  [Q] quit                                  |' -ForegroundColor DarkGray
    Write-Host '+----------------------------------------------------------------------+' -ForegroundColor Cyan
}

$paused = $false
try {
    while ($true) {
        if (-not $paused) { Draw }
        $deadline = (Get-Date).AddSeconds($RefreshSeconds)
        while ((Get-Date) -lt $deadline) {
            $hasKey = $false
            try { $hasKey = [Console]::KeyAvailable } catch { $hasKey = $false }
            if ($hasKey) {
                $key = [Console]::ReadKey($true).Key
                if ($key -eq 'Q' -or $key -eq 'Escape') { return }
                if ($key -eq 'P') { $paused = -not $paused; break }
                if ($key -eq 'R') { break }
            }
            Start-Sleep -Milliseconds 100
        }
    }
}
catch {
    Write-Host ("Monitor error: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Presiona Enter para cerrar.' -ForegroundColor DarkGray
    [void](Read-Host)
}
finally {
    Clear-Host
    Write-Host 'Monitor chat cerrado.' -ForegroundColor DarkGray
}
