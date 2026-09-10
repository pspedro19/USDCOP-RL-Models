param(
    [int]$DurationMinutes = 240,
    [int]$PollSeconds = 10
)

$coordinationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$monitorRoot = Join-Path $coordinationRoot "monitor"
$logPath = Join-Path $monitorRoot "CODEX-MESSAGES.log"
$pidPath = Join-Path $monitorRoot "CODEX-MESSAGES.pid"
$targets = @(
    (Join-Path $coordinationRoot "INBOX-CODEX.md"),
    (Join-Path $coordinationRoot "CONTRACTS.md"),
    (Join-Path $coordinationRoot "CLAUDE-STATUS.md")
)

New-Item -ItemType Directory -Path $monitorRoot -Force | Out-Null
$deadline = (Get-Date).AddMinutes($DurationMinutes)
$state = @{}

function Get-ContentHash([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return "ABSENT" }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

foreach ($target in $targets) {
    $state[$target] = Get-ContentHash $target
}

Set-Content -LiteralPath $pidPath -Encoding ascii -Value $PID
Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
    "[{0:o}] START pid={1} duration_minutes={2} poll_seconds={3} mode=sha256" -f
    (Get-Date).ToUniversalTime(), $PID, $DurationMinutes, $PollSeconds
)

try {
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds $PollSeconds
        foreach ($target in $targets) {
            $digest = Get-ContentHash $target
            if ($state[$target] -eq $digest) { continue }
            $previous = $state[$target]
            $state[$target] = $digest
            $lastMessage = ""
            if ((Split-Path -Leaf $target) -eq "INBOX-CODEX.md") {
                $match = Select-String -LiteralPath $target -Pattern '^\[CLD-[0-9]+' |
                    Select-Object -Last 1
                if ($null -ne $match) { $lastMessage = $match.Line }
            }
            Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
                "[{0:o}] CHANGE file={1} old={2} new={3} last={4}" -f
                (Get-Date).ToUniversalTime(), (Split-Path -Leaf $target),
                $previous.Substring(0, [Math]::Min(12, $previous.Length)),
                $digest.Substring(0, [Math]::Min(12, $digest.Length)), $lastMessage
            )
        }
    }
} finally {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        "[{0:o}] STOP pid={1}" -f (Get-Date).ToUniversalTime(), $PID
    )
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
