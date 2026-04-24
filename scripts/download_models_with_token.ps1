# Descarga de modelos usando huggingface-cli (con token)
# Requiere: pip install huggingface_hub
# Configura tu token en la variable HF_TOKEN

param(
    [string]$Token = ""
)

if ([string]::IsNullOrWhiteSpace($Token)) {
    Write-Host ""
    Write-Host "Necesitas un token de HuggingFace." -ForegroundColor Yellow
    Write-Host "1. Ve a: https://huggingface.co/settings/tokens"
    Write-Host "2. Crea un token (rol 'read')"
    Write-Host "3. Péguelo a continuación."
    Write-Host ""
    $Token = Read-Host "Token de HuggingFace (hf_XXXXX...)"
}

$base = Resolve-Path (Join-Path $PSScriptRoot "..\ui\tob\ComfyUI\models")

Write-Host ""
Write-Host "=== Descargando modelos con token ===" -ForegroundColor Cyan

function Download-HF {
    param($RepoId, $Filename, $OutDir, $Label)
    Ensure-Dir $OutDir
    $outfile = Join-Path $OutDir $Filename

    Write-Host "`n[$Label] Descargando..." -ForegroundColor Yellow
    Write-Host "  Repo: $RepoId"
    Write-Host "  Archivo: $Filename"

    try {
        huggingface-cli download $RepoId $Filename --local-dir $OutDir --token $Token
        Write-Host "  ✅ OK" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Ensure-Dir {
    param([string]$path)
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# 1. Wan 2.2
Download-HF "Comfy-UI/Wan2.2-I2V-14B-480P-Q4_K_M" "Wan2.2-I2V-14B-480P-Q4_K_M.safetensors" "$base\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M" "Wan 2.2 (8GB)"
# VAE Wan
Download-HF "Comfy-UI/Wan2.2-VAE" "Wan2.2_VAE.safetensors" "$base\vae" "Wan VAE (50MB)"
# LTX Video
Download-HF "Comfy-UI/LTX-Video-0.9.5" "ltxvideo.safetensors" "$base\checkpoints\ltx-video-0.9.5" "LTX Video 0.9.5 (6GB)"

Write-Host ""
Write-Host "=== DESCARGA COMPLETADA ===" -ForegroundColor Green
Write-Host ""
Write-Host "Pasos siguientes:"
Write-Host "1. Reinicia la aplicación: python run.py"
Write-Host "2. Ve a la pestaña 'Imagine'"
Write-Host "3. Pulsa '🔍 Detectar Modelos'"
Write-Host ""
Pause
