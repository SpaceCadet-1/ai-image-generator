# start.ps1 - Start backend + frontend
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "  AI Image Generator - Starting" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Start FastAPI backend (port 3001)
$api = Start-Job -Name "API" -ScriptBlock {
    param($dir)
    Set-Location $dir
    & "$dir\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 3001 2>&1
} -ArgumentList "$root\server"

# Start Vite frontend (port 5173)
$web = Start-Job -Name "WEB" -ScriptBlock {
    param($dir)
    Set-Location $dir
    & npm run dev 2>&1
} -ArgumentList "$root\client"

$jobs = @($api, $web)

Write-Host "[API] FastAPI server starting on http://localhost:3001" -ForegroundColor Green
Write-Host "[WEB] Vite dev server starting on http://localhost:5173" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

$colors = @{ "API" = "Green"; "WEB" = "Magenta" }

try {
    [Console]::TreatControlCAsInput = $true
} catch {}

try {
    while ($true) {
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if ($key.Key -eq 'C' -and $key.Modifiers -eq 'Control') { break }
        }

        $anyRunning = $false
        foreach ($job in $jobs) {
            if ($job.HasMoreData) {
                $output = Receive-Job -Job $job -ErrorAction SilentlyContinue
                foreach ($line in $output) {
                    $text = "$line"
                    if ($text.Trim() -ne "") {
                        Write-Host "[$($job.Name)] " -ForegroundColor $colors[$job.Name] -NoNewline
                        Write-Host $text
                    }
                }
            }
            if ($job.State -eq 'Running') { $anyRunning = $true }
            elseif ($job.State -eq 'Failed') {
                Write-Host "[$($job.Name)] Job failed!" -ForegroundColor Red
            }
        }

        if (-not $anyRunning) {
            Write-Host "All jobs stopped." -ForegroundColor Red
            break
        }

        Start-Sleep -Milliseconds 200
    }
}
finally {
    try { [Console]::TreatControlCAsInput = $false } catch {}
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Red
    foreach ($job in $jobs) {
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Done." -ForegroundColor Red
}
