# Copy all torch lib files
$src = "C:\Users\INdaHouse\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
$dst = "D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torch\lib"

# Remove existing files first
Get-ChildItem $dst -File | Remove-Item -Force -ErrorAction SilentlyContinue

# Copy all files from source
Copy-Item "$src\*" -Destination $dst -Force

Write-Host "Done. Files count:"
Get-ChildItem $dst | Measure-Object | Select-Object -ExpandProperty Count
