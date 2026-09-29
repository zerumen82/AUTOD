$ErrorActionPreference = 'Stop'
$outputDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    'ERROR: este proceso no tiene permisos de administrador.' | Set-Content -LiteralPath (Join-Path $outputDir 'chkdsk_status.txt') -Encoding UTF8
    exit 5
}

$results = foreach ($drive in @('C:', 'D:')) {
    $letter = $drive.Substring(0, 1)
    $logPath = Join-Path $outputDir "chkdsk_${letter}_scan.txt"
    & chkdsk.exe $drive /scan *> $logPath
    [pscustomobject]@{
        Drive = $drive
        ExitCode = $LASTEXITCODE
        Log = $logPath
    }
}
$results | Format-Table -AutoSize | Out-String | Set-Content -LiteralPath (Join-Path $outputDir 'chkdsk_status.txt') -Encoding UTF8
