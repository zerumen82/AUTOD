$orig = "C:\Users\INdaHouse\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib\shm.dll"
$venv = "D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torch\lib\shm.dll"
$fresh = "D:\tmp\torch_extracted\torch\lib\shm.dll"

$o = Get-Item $orig
$v = Get-Item $venv  
$f = Get-Item $fresh

Write-Host "Original:" $o.Length $o.LastWriteTime
Write-Host "Venv:" $v.Length $v.LastWriteTime
Write-Host "Fresh:" $f.Length $f.LastWriteTime

# Check file hash
$oHash = (Get-FileHash $orig -Algorithm MD5).Hash
$vHash = (Get-FileHash $venv -Algorithm MD5).Hash
$fHash = (Get-FileHash $fresh -Algorithm MD5).Hash

Write-Host "Original hash:" $oHash
Write-Host "Venv hash:" $vHash
Write-Host "Fresh hash:" $fHash
