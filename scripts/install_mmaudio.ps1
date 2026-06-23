# Instala ComfyUI-MMAudio + modelos FP16 para Animate Image
# Uso: powershell -ExecutionPolicy Bypass -File scripts/install_mmaudio.ps1

$Host.UI.RawUI.WindowTitle = "Instalar MMAudio - AutoAuto"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$comfy = Join-Path $root "ui\tob\ComfyUI"
$customNodes = Join-Path $comfy "custom_nodes"
$mmaudioNode = Join-Path $customNodes "ComfyUI-MMAudio"
$modelsDir = Join-Path $comfy "models\mmaudio"
$hfBase = "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main"

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

function Download-File($url, $outFile, $label) {
    Ensure-Dir (Split-Path $outFile -Parent)
    if (Test-Path $outFile) {
        Write-Host "Ya existe: $label" -ForegroundColor DarkGray
        return $true
    }
    Write-Host "Descargando: $label" -ForegroundColor Green
    try {
        Invoke-WebRequest -Uri $url -OutFile $outFile -UserAgent "Mozilla/5.0"
        Write-Host "OK: $label" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "Error: $label - $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Instalacion MMAudio (Animate Image)" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $comfy)) {
    Write-Host "No se encontro ComfyUI en: $comfy" -ForegroundColor Red
    exit 1
}

Ensure-Dir $customNodes
Ensure-Dir $modelsDir

if (-not (Test-Path (Join-Path $mmaudioNode "nodes.py"))) {
    Write-Host "Clonando ComfyUI-MMAudio..." -ForegroundColor Yellow
    git clone https://github.com/kijai/ComfyUI-MMAudio.git $mmaudioNode
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Fallo git clone. Comprueba que git este instalado." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ComfyUI-MMAudio ya instalado." -ForegroundColor DarkGray
}

$reqFile = Join-Path $mmaudioNode "requirements.txt"
if (Test-Path $reqFile) {
    Write-Host "Instalando dependencias Python (venv ComfyUI)..." -ForegroundColor Yellow
    $comfyPython = Join-Path $comfy "venv\Scripts\python.exe"
    $projectPython = Join-Path $root "venv\Scripts\python.exe"
    if (Test-Path $comfyPython) {
        & $comfyPython -m pip install -r $reqFile
    } elseif (Test-Path $projectPython) {
        & $projectPython -m pip install -r $reqFile
    } else {
        pip install -r $reqFile
    }
}

$models = @(
    @{ Name = "mmaudio_large_44k_v2_fp16.safetensors"; Size = "~2 GB" },
    @{ Name = "mmaudio_vae_44k_fp16.safetensors"; Size = "~611 MB" },
    @{ Name = "mmaudio_synchformer_fp16.safetensors"; Size = "~475 MB" },
    @{ Name = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"; Size = "~1.9 GB" }
)

$ok = $true
foreach ($m in $models) {
    $dest = Join-Path $modelsDir $m.Name
    if (-not (Download-File "$hfBase/$($m.Name)" $dest "$($m.Name) $($m.Size)")) {
        $ok = $false
    }
}

Write-Host ""
if ($ok) {
    Write-Host "MMAudio instalado correctamente." -ForegroundColor Green
    Write-Host "Reinicia ComfyUI y luego prueba Animate Image con 'Anadir audio (MMAudio)'." -ForegroundColor Cyan
} else {
    Write-Host "Instalacion incompleta. Revisa errores de descarga arriba." -ForegroundColor Yellow
    exit 1
}