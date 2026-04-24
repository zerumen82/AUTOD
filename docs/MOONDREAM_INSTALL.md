# Guía de instalación Moondream 2

Moondream 2 es un pequeño modelo multimodal diseñado específicamente para describir imágenes y generar prompts perfectos para FLUX y modelos de difusión.

## 📥 Descarga del modelo

Descarga el modelo Q4_K_M (1.8GB, recomendado para 8GB VRAM):

```powershell
cd D:\PROJECTS\AUTOAUTO
mkdir models
cd models

# Descarga con wget o manualmente desde HuggingFace
$url = "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.Q4_K_M.gguf"
$dest = "moondream2-text-model-f16.Q4_K_M.gguf"

Invoke-WebRequest -Uri $url -OutFile $dest
```

URL directa: https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.Q4_K_M.gguf

## 📦 Instalación de dependencias

```powershell
cd D:\PROJECTS\AUTOAUTO
venv\Scripts\python.exe -m pip install llama-cpp-python
```

## ✅ Verificación

Coloca el modelo en:
```
D:\PROJECTS\AUTOAUTO\models\moondream2-text-model-f16.Q4_K_M.gguf
```

## 🎯 Uso

El botón **🔍 Analizar Imagen** en ImageEditor usará automáticamente Moondream 2 si el modelo está presente. Si no lo encuentra, usará el analizador básico como fallback.

## 🚀 Características:

✅ Detecta:
- ✅ Número exacto de personas, género, edad
- ✅ Poses, expresiones, ropa, objetos
- ✅ Tipo de escena, iluminación, ambiente
- ✅ Calidad de la foto, detalles
- ✅ Genera prompts listos para usar directamente en FLUX
- ✅ 100% sin censura
- ✅ ~1 segundo por descripción
- ✅ Utiliza CUDA automáticamente si está disponible
