# start.ps1 - Start all 3 servers (GPU, API, Web)
# Usage: .\start.ps1
# Press Ctrl+C to stop all servers

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI Image Generator - Starting All..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = $PSScriptRoot

# Start FastAPI (GPU/SDXL) server
$gpuJob = Start-Job -Name "GPU" -ScriptBlock {
    param($dir)
    Set-Location $dir
    & "$dir\.venv\Scripts\python.exe" run.py 2>&1
} -ArgumentList "$root\local-server"

# Start Express (API) server
$apiJob = Start-Job -Name "API" -ScriptBlock {
    param($dir)
    Set-Location $dir
    & node index.js 2>&1
} -ArgumentList "$root\server"

# Start React/Vite (Web) dev server
$webJob = Start-Job -Name "WEB" -ScriptBlock {
    param($dir)
    Set-Location $dir
    & npm run dev 2>&1
} -ArgumentList "$root\client"

$jobs = @($gpuJob, $apiJob, $webJob)

Write-Host "[GPU] FastAPI/SDXL server starting on port 8000" -ForegroundColor Yellow
Write-Host "[API] Express server starting on port 3001" -ForegroundColor Green
Write-Host "[WEB] Vite dev server starting" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press Ctrl+C to stop all servers." -ForegroundColor DarkGray
Write-Host ""

# Cleanup function
function Stop-AllJobs {
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Red

    foreach ($job in $jobs) {
        if ($job.State -ne 'Completed' -and $job.State -ne 'Failed') {
            Stop-Job -Job $job -ErrorAction SilentlyContinue
        }
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }

    Write-Host "All servers stopped." -ForegroundColor Red
}

# Register Ctrl+C handler
try {
    [Console]::TreatControlCAsInput = $true
} catch {
    # Fallback: TreatControlCAsInput may not be available in all hosts
}

# Color map for job prefixes
$colors = @{
    "GPU" = "Yellow"
    "API" = "Green"
    "WEB" = "Magenta"
}

# Tail output from all jobs
try {
    while ($true) {
        # Check for Ctrl+C
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if ($key.Key -eq 'C' -and $key.Modifiers -eq 'Control') {
                break
            }
        }

        $anyRunning = $false

        foreach ($job in $jobs) {
            $name = $job.Name
            $color = $colors[$name]

            if ($job.HasMoreData) {
                $output = Receive-Job -Job $job -ErrorAction SilentlyContinue
                foreach ($line in $output) {
                    $text = "$line"
                    if ($text.Trim() -ne "") {
                        Write-Host "[$name] " -ForegroundColor $color -NoNewline
                        Write-Host $text
                    }
                }
            }

            if ($job.State -eq 'Running') {
                $anyRunning = $true
            }
            elseif ($job.State -eq 'Failed') {
                Write-Host "[$name] Job failed!" -ForegroundColor Red
            }
        }

        if (-not $anyRunning) {
            Write-Host "All jobs have stopped." -ForegroundColor Red
            break
        }

        Start-Sleep -Milliseconds 200
    }
}
finally {
    try {
        [Console]::TreatControlCAsInput = $false
    } catch {}
    Stop-AllJobs
}
