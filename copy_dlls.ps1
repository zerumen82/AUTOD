$src = "C:\Users\INdaHouse\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
$dst = "D:\PROJECTS\AUTOAUTO\venv_flux\Lib\site-packages\torch\lib"

Copy-Item "$src\*" -Destination $dst -Force -ErrorAction SilentlyContinue
Write-Host "DLLs copied"