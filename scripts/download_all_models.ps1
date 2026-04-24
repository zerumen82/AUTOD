# Descarga directa de modelos (sin interacción)
# Ejecuta: powershell -ExecutionPolicy Bypass -File .\download_all_models.ps1

$base = Resolve-Path (Join-Path $PSScriptRoot "..\ui\tob\ComfyUI\models")

Write-Host ""
Write-Host "=== Descargando modelos para Animate Image ===" -ForegroundColor Cyan
Write-Host ""

# Función de descarga con barra de progreso
function Download-Model {
    param($Url, $OutFile, $Label)
    $dir = Split-Path $OutFile -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    Write-Host "`n[$($Label)] Descargando..." -ForegroundColor Yellow
    Write-Host "  Ruta: $OutFile"

    try {
        # Usar Invoke-WebRequest con UserAgent y timeout
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UserAgent "Mozilla/5.0" -TimeoutSec 300
        Write-Host "  ✅ OK" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 1. Wan 2.2 + VAE
$wan_url = "https://huggingface.co/Comfy-UI/Wan2.2-I2V-14B-480P-Q4_K_M/resolve/main/Wan2.2-I2V-14B-480P-Q4_K_M.safetensors"
$wan_out = "$base\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M\Wan2.2-I2V-14B-480P-Q4_K_M.safetensors"
$vae_url = "https://huggingface.co/Comfy-UI/Wan2.2-VAE/resolve/main/Wan2.2_VAE.safetensors"
$vae_out = "$base\vae\Wan2.2_VAE.safetensors"

Download-Model $wan_url $wan_out "Wan 2.2 (8GB)"
Download-Model $vae_url $vae_out "Wan VAE (50MB)"

# 2. LTX Video 0.9.5
$ltx_url = "https://huggingface.co/Comfy-UI/LTX-Video-0.9.5/resolve/main/ltxvideo.safetensors"
$ltx_out = "$base\checkpoints\ltx-video-0.9.5\ltxvideo.safetensors"
Download-Model $ltx_url $ltx_out "LTX Video 0.9.5 (6GB)"

Write-Host ""
Write-Host "=== DESCarga COMPLETADA ===" -ForegroundColor Green
Write-Host ""
Write-Host "Pasos siguientes:"
Write-Host "1. Reinicia la aplicación: python run.py"
Write-Host "2. Ve a la pestaña 'Imagine'"
Write-Host "3. Pulsa '🔍 Detectar Modelos'"
Write-Host ""
