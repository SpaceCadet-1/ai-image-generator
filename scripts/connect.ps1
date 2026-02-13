# connect.ps1 - Tunnel to AWS instance via SSM (no SSH needed)
#
# Prerequisites:
#   1. AWS CLI v2: https://aws.amazon.com/cli/
#   2. Session Manager plugin
#   3. AWS SSO configured: aws configure sso
#
# Usage:
#   .\scripts\connect.ps1 i-0abc123def456
#   .\scripts\connect.ps1 -InstanceId i-0abc123def456 -Profile SysmexAI
#
# Then open http://localhost:3001 in your browser.

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

Write-Host "Starting tunnel: localhost:3001 -> instance:3001" -ForegroundColor Green
Write-Host ""
Write-Host "Open http://localhost:3001 in your browser." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to disconnect." -ForegroundColor DarkGray
Write-Host ""

# Run tunnel in foreground (simpler, more reliable)
aws ssm start-session `
    --target $InstanceId `
    --document-name AWS-StartPortForwardingSession `
    --parameters "portNumber=3001,localPortNumber=3001" `
    --profile $Profile `
    --region $Region
