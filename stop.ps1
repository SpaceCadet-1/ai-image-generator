# stop.ps1 - Force-stop backend + frontend processes
# Usage: .\stop.ps1

Write-Host ""
Write-Host "Stopping AI Image Generator..." -ForegroundColor Red
Write-Host ""

$stopped = 0

# Kill processes on port 3001 (FastAPI backend)
$apiPids = Get-NetTCPConnection -LocalPort 3001 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid in $apiPids) {
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "[API] Killing $($proc.ProcessName) (PID $pid) on port 3001" -ForegroundColor Green
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        $stopped++
    }
}

# Kill processes on port 5173 (Vite frontend)
$webPids = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid in $webPids) {
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "[WEB] Killing $($proc.ProcessName) (PID $pid) on port 5173" -ForegroundColor Magenta
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        $stopped++
    }
}

# Clean up PowerShell background jobs
Get-Job -ErrorAction SilentlyContinue | Where-Object { $_.Name -in @("API", "WEB") } | ForEach-Object {
    Write-Host "Removing job: $($_.Name) ($($_.State))" -ForegroundColor DarkGray
    Stop-Job $_ -ErrorAction SilentlyContinue
    Remove-Job $_ -Force -ErrorAction SilentlyContinue
    $stopped++
}

if ($stopped -eq 0) {
    Write-Host "No running processes found." -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "Stopped $stopped process(es)." -ForegroundColor Red
}
Write-Host ""
