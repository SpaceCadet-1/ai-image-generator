# stop.ps1 - Force-stop all project-related processes
# Usage: .\stop.ps1

Write-Host ""
Write-Host "Stopping AI Image Generator processes..." -ForegroundColor Red
Write-Host ""

$stopped = 0

# Kill Python processes running on port 8000 (FastAPI/uvicorn)
$pythonProcs = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid in $pythonProcs) {
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "[GPU] Killing $($proc.ProcessName) (PID $pid) on port 8000" -ForegroundColor Yellow
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        $stopped++
    }
}

# Kill Node processes running on port 3001 (Express)
$nodeApiProcs = Get-NetTCPConnection -LocalPort 3001 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid in $nodeApiProcs) {
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "[API] Killing $($proc.ProcessName) (PID $pid) on port 3001" -ForegroundColor Green
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        $stopped++
    }
}

# Kill Node processes running on port 5173 (Vite default)
$nodeWebProcs = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid in $nodeWebProcs) {
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "[WEB] Killing $($proc.ProcessName) (PID $pid) on port 5173" -ForegroundColor Magenta
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        $stopped++
    }
}

# Also clean up any lingering PowerShell background jobs
$bgJobs = Get-Job -ErrorAction SilentlyContinue | Where-Object { $_.Name -in @("GPU", "API", "WEB") }
foreach ($job in $bgJobs) {
    Write-Host "Removing background job: $($job.Name) ($($job.State))" -ForegroundColor DarkGray
    Stop-Job -Job $job -ErrorAction SilentlyContinue
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    $stopped++
}

if ($stopped -eq 0) {
    Write-Host "No running processes found." -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "Done. Stopped $stopped process(es)." -ForegroundColor Red
}

Write-Host ""
