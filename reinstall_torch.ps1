# Reinstall torch from fresh wheel
$venv = "D:\PROJECTS\AUTOAUTO\venv\Scripts\pip"

# Uninstall
& $venv uninstall torch torchvision torchaudio -y 2>$null

# Clean torch directories
$dirs = @(
    "D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torch",
    "D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torchvision",
    "D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torchaudio"
)

foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force
    }
}

Write-Host "Cleaned torch"

# Install fresh
& $venv install torch==2.4.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Done"
