# Plan de instalación: Video con Audio para RTX 3060 Ti 8GB

## Modelos a descargar

### 1. Stable Diffusion 1.5 (para generar imágenes desde prompt)
- **Archivo**: `v1-5-pruned.safetensors`
- **Tamaño**: ~4GB
- **Enlace**: https://huggingface.co/runwayml/stable-diffusion-v1-5
- **Ubicación**: `ui/tob/ComfyUI/models/checkpoints/`

### 2. Stable Video Diffusion (SVD)
- **Archivo**: `svd_xt.safetensors`
- **Tamaño**: ~5GB
- **Enlace**: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
- **Ubicación**: `ui/tob/ComfyUI/models/checkpoints/`

## Workflow completo

El workflow hace:
1. Prompt de texto → Generar imagen (SD 1.5)
2. Imagen → Convertir a video (SVD)
3. Generar archivo batch para añadir audio

## Comandos para añadir audio

```bash
# Instalación de ffmpeg (ya está en tools/)
# Usar para añadir audio al video:

ffmpeg -i "video_sin_audio.mp4" -i "audio.mp3" -c:v copy -c:a aac "video_final.mp4"
```

## Para más calidad de audio

Puedes usar servicios de text-to-speech como:
- Azure Speech
- ElevenLabs (mejor calidad)
- OpenAI TTS

Luego combinas el video generado con el audio.
