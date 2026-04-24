# Descarga de Modelos para Animate Image
# Ejecuta: powershell -ExecutionPolicy Bypass -File download_animate_models.ps1

$Host.UI.RawUI.WindowTitle = "Descarga de Modelos - AutoAuto"

function Show-Header {
    Write-Host ""
    Write-Host "====================================" -ForegroundColor Cyan
    Write-Host "  Descarga de Modelos para Animate Image" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Menu {
    Write-Host "Selecciona los modelos a descargar:" -ForegroundColor Yellow
    Write-Host "1) Wan 2.2 (14B GGUF) - Máxima calidad (~8GB) + VAE"
    Write-Host "2) LTX Video 0.9.5 - Velocidad (~6GB)"
    Write-Host "3) Ambos (Wan + LTX)"
    Write-Host ""
}

function Ensure-Dir {
    param([string]$path)
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Label
    )
    $dir = Split-Path $OutFile -Parent
    Ensure-Dir $dir

    Write-Host "`nDescargando: $Label" -ForegroundColor Green
    Write-Host "  Destino: $OutFile"

    try {
        # Usar Invoke-WebRequest con progreso nativo de PowerShell
        $ProgressPreference = 'Continue'
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UserAgent "Mozilla/5.0"
        Write-Host "✅ Completado: $Label" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ Error descargando $Label" -ForegroundColor Red
        Write-Host "   $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

# MAIN
Show-Header
Show-Menu

$choice = Read-Host "Opción (1-3)"

$base = Resolve-Path (Join-Path $PSScriptRoot "..\ui\tob\ComfyUI\models")

switch ($choice) {
    "1" {
        Download-File "https://huggingface.co/Comfy-UI/Wan2.2-I2V-14B-480P-Q4_K_M/resolve/main/Wan2.2-I2V-14B-480P-Q4_K_M.safetensors" "$base\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M\Wan2.2-I2V-14B-480P-Q4_K_M.safetensors" "Wan 2.2 (8GB)"
        Download-File "https://huggingface.co/Comfy-UI/Wan2.2-VAE/resolve/main/Wan2.2_VAE.safetensors" "$base\vae\Wan2.2_VAE.safetensors" "Wan VAE (50MB)"
        break
    }
    "2" {
        Download-File "https://huggingface.co/Comfy-UI/LTX-Video-0.9.5/resolve/main/ltxvideo.safetensors" "$base\checkpoints\ltx-video-0.9.5\ltxvideo.safetensors" "LTX Video 0.9.5 (6GB)"
        break
    }
    "3" {
        Download-File "https://huggingface.co/Comfy-UI/Wan2.2-I2V-14B-480P-Q4_K_M/resolve/main/Wan2.2-I2V-14B-480P-Q4_K_M.safetensors" "$base\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M\Wan2.2-I2V-14B-480P-Q4_K_M.safetensors" "Wan 2.2 (8GB)"
        Download-File "https://huggingface.co/Comfy-UI/Wan2.2-VAE/resolve/main/Wan2.2_VAE.safetensors" "$base\vae\Wan2.2_VAE.safetensors" "Wan VAE (50MB)"
        Download-File "https://huggingface.co/Comfy-UI/LTX-Video-0.9.5/resolve/main/ltxvideo.safetensors" "$base\checkpoints\ltx-video-0.9.5\ltxvideo.safetensors" "LTX Video 0.9.5 (6GB)"
        break
    }
    default {
        Write-Host "Opción no válida." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "INSTALACIÓN COMPLETADA" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pasos siguientes:"
Write-Host "1. Reinicia la aplicación: python run.py"
Write-Host "2. Ve a la pestaña 'Imagine'"
Write-Host "3. Pulsa '🔍 Detectar Modelos'"
Write-Host ""
Write-Host "Nota: Si la descarga falla (timeout), vuelve a ejecutar el script."
Write-Host ""
Pause
