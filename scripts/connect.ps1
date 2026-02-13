# connect.ps1 - Tunnel to AWS instance via SSM (no SSH needed)
#
# Prerequisites:
#   1. AWS CLI v2: https://aws.amazon.com/cli/
#   2. Session Manager plugin: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html
#   3. AWS SSO configured: aws configure sso
#
# Usage:
#   .\scripts\connect.ps1 -InstanceId i-0abc123def456 -Profile SysmexAI
#   .\scripts\connect.ps1 i-0abc123def456 SysmexAI
#
# Then open http://localhost:3000 in your browser.

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$InstanceId,

    [Parameter(Position=1)]
    [string]$Profile = "SysmexAI",

    [Parameter()]
    [string]$Region = "us-east-1"
)

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "  Connecting to AWS instance" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Instance: $InstanceId" -ForegroundColor Yellow
Write-Host "Profile:  $Profile" -ForegroundColor Yellow
Write-Host "Region:   $Region" -ForegroundColor Yellow
Write-Host ""

# Check prerequisites
$missing = @()
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { $missing += "AWS CLI (https://aws.amazon.com/cli/)" }
if (-not (Get-Command session-manager-plugin -ErrorAction SilentlyContinue)) {
    $missing += "Session Manager plugin (https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)"
}
if ($missing.Count -gt 0) {
    Write-Host "Missing prerequisites:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    exit 1
}

# Start both tunnels as background jobs
Write-Host "Starting port tunnels..." -ForegroundColor Green
Write-Host "  localhost:3001 -> instance:3001 (FastAPI backend)" -ForegroundColor Green
Write-Host "  localhost:3000 -> instance:3000 (Vite frontend)" -ForegroundColor Magenta
Write-Host ""

$apiTunnel = Start-Job -Name "SSM-API" -ScriptBlock {
    param($id, $prof, $reg)
    aws ssm start-session `
        --target $id `
        --document-name AWS-StartPortForwardingSession `
        --parameters "portNumber=3001,localPortNumber=3001" `
        --profile $prof `
        --region $reg 2>&1
} -ArgumentList $InstanceId, $Profile, $Region

$webTunnel = Start-Job -Name "SSM-WEB" -ScriptBlock {
    param($id, $prof, $reg)
    aws ssm start-session `
        --target $id `
        --document-name AWS-StartPortForwardingSession `
        --parameters "portNumber=3000,localPortNumber=3000" `
        --profile $prof `
        --region $reg 2>&1
} -ArgumentList $InstanceId, $Profile, $Region

# Wait a moment for tunnels to establish
Start-Sleep -Seconds 3

# Check tunnel status
$apiState = (Get-Job -Name "SSM-API").State
$webState = (Get-Job -Name "SSM-WEB").State

if ($apiState -eq "Running" -and $webState -eq "Running") {
    Write-Host "Tunnels active!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Open http://localhost:3000 in your browser." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to disconnect." -ForegroundColor DarkGray
    Write-Host ""
} else {
    Write-Host "Warning: tunnel state - API=$apiState, WEB=$webState" -ForegroundColor Yellow
    # Show any errors
    Receive-Job -Name "SSM-API" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[API] $_" -ForegroundColor Yellow }
    Receive-Job -Name "SSM-WEB" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[WEB] $_" -ForegroundColor Yellow }
}

# Keep running, stream output, until Ctrl+C
try {
    while ($true) {
        foreach ($name in @("SSM-API", "SSM-WEB")) {
            $job = Get-Job -Name $name -ErrorAction SilentlyContinue
            if ($job -and $job.HasMoreData) {
                Receive-Job -Job $job -ErrorAction SilentlyContinue | ForEach-Object {
                    $text = "$_"
                    if ($text.Trim() -ne "") { Write-Host "[$name] $text" -ForegroundColor DarkGray }
                }
            }
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Write-Host ""
    Write-Host "Closing tunnels..." -ForegroundColor Red
    Get-Job -Name "SSM-API" -ErrorAction SilentlyContinue | Stop-Job -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
    Get-Job -Name "SSM-WEB" -ErrorAction SilentlyContinue | Stop-Job -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
    Get-Job | Where-Object { $_.Name -like "SSM-*" } | Stop-Job -ErrorAction SilentlyContinue
    Get-Job | Where-Object { $_.Name -like "SSM-*" } | Remove-Job -Force -ErrorAction SilentlyContinue
    Write-Host "Disconnected." -ForegroundColor Red
}
