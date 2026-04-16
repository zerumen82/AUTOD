$src = 'C:\Users\INdaHouse\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib'
$dst = 'D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torch\lib'

Get-ChildItem $src -Filter '*.dll' | ForEach-Object {
    Copy-Item $_.FullName -Destination $dst -Force
    Write-Host "Copied: $($_.Name)"
}
Write-Host 'Done'
