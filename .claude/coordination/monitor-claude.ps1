param(
    [int]$DurationMinutes = 240,
    [int]$PollSeconds = 10
)

$coordinationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$monitorRoot = Join-Path $coordinationRoot "monitor"
$logPath = Join-Path $monitorRoot "CLAUDE-MESSAGES.log"
$targets = @(
    (Join-Path $coordinationRoot "INBOX-CODEX.md"),
    (Join-Path $coordinationRoot "INBOX-CLAUDE.md"),
    (Join-Path $coordinationRoot "CLAUDE-STATUS.md"),
    (Join-Path $coordinationRoot "CODEX-STATUS.md")
)

New-Item -ItemType Directory -Path $monitorRoot -Force | Out-Null
$deadline = (Get-Date).AddMinutes($DurationMinutes)
$state = @{}

foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target) {
        $item = Get-Item -LiteralPath $target
        $state[$target] = "$($item.LastWriteTimeUtc.Ticks):$($item.Length)"
    }
}

Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
    "[{0:o}] START pid={1} duration_minutes={2} poll_seconds={3}" -f
    (Get-Date).ToUniversalTime(), $PID, $DurationMinutes, $PollSeconds
)

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds $PollSeconds
    foreach ($target in $targets) {
        if (-not (Test-Path -LiteralPath $target)) {
            continue
        }
        $item = Get-Item -LiteralPath $target
        $fingerprint = "$($item.LastWriteTimeUtc.Ticks):$($item.Length)"
        if ($state[$target] -eq $fingerprint) {
            continue
        }
        $state[$target] = $fingerprint
        $lastMessage = ""
        if ($item.Name -like "INBOX-*.md") {
            $match = Select-String -LiteralPath $target -Pattern '^\- \[(CLD|CXD|MSG)-' |
                Select-Object -Last 1
            if ($null -ne $match) {
                $lastMessage = $match.Line
            }
        }
        Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
            "[{0:o}] CHANGE file={1} bytes={2} last={3}" -f
            (Get-Date).ToUniversalTime(), $item.Name, $item.Length, $lastMessage
        )
    }
}

Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
    "[{0:o}] STOP pid={1}" -f (Get-Date).ToUniversalTime(), $PID
)
