# Guía Rápida: Obtener LoRAs NSFW para FLUX.2-klein-4B

## 🚀 Opción Más Simple: Usar Modelo Base (RECOMENDADO)

El modelo base `FLUX.2-klein-4B` con el text encoder uncensored **YA GENERA NSFW** sin necesidad de LoRA.

**Pasos**:
1. No selecciones ningún LoRA en la UI (deja el dropdown en "None" o el LoRA por defecto)
2. Usa prompts explícitos:
   ```
   nude, naked, bare breasts, explicit nsfw content, adult, erotic
   ```
3. El text encoder `qwen3-4b-abl-q4_0.gguf` está configurado para no censurar.

**Resultado**: Funciona ✅, sin errores de compatibilidad.

---

## 📥 Opción 2: Descargar LoRA Compatible (público)

### Script Automático

Ejecuta el script que ha sido creado:

```powershell
cd D:\PROJECTS\AUTOAUTO
python scripts\download_flux_klein_lora.py
```

El script te mostrará un menú de LoRAs compatibles conocidos para descargar directamente.

---

### Descarga Manual (por consola)

Si prefieres línea de comandos:

#### **Opción A: Con login en HuggingFace** (para LoRAs privados/gated)
```powershell
# 1. Login (una sola vez)
huggingface-cli login
# Pegar tu token desde https://huggingface.co/settings/tokens

# 2. Descargar un LoRA específico
huggingface-cli download DeverStyle/Flux.2-Klein-Loras `
  alba_baptista_vrtlalbabaptista_flux2_klein_4b.safetensors `
  --local-dir "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras" `
  --local-dir-use-symlinks false
```

#### **Opción B: Descarga directa pública** (sin login)
```powershell
# Ejemplo: LoRA "zoom" (45MB, público)
$url = "https://huggingface.co/fal/flux-2-klein-4B-zoom-lora/resolve/main/flux-2-klein-4b-zoom-lora.safetensors"
$dest = "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras\flux-2-klein-4b-zoom-lora.safetensors"

Invoke-WebRequest -Uri $url -OutFile $dest
```

---

## 🔍 Cómo Identificar un LoRA Compatible

**REQUISITOS para FLUX.2-klein-4B**:

| Característica | Valor correcto | Valor incorrecto |
|----------------|----------------|------------------|
| Tamaño archivo | 100–350 MB | >400 MB (son para 9B) |
| Nombre contiene | `klein`, `flux2_klein`, `f2klein` | `dev`, `schnell`, `flux1_`, `flux-1` |
| Base model | `FLUX.2-klein-4B` o `flux_klein` | `FLUX.1-dev`, `flux1`, `schnell` |

**Fórmula rápida**:
- ✅ **Válido**: `*klein*.safetensors` (150-300MB)
- ❌ **Inválido**: `*dev*.safetensors` (>400MB) o `*schnell*.safetensors`

---

## 📋 Lista de LoRAs Verificados (Públicos/Accesibles)

| Nombre | Repositorio | Tamaño | NSFW | Login |
|--------|-------------|--------|------|-------|
| `alba_baptista_vrtlalbabaptista_flux2_klein_4b.safetensors` | DeverStyle/Flux.2-Klein-Loras | 158MB | Sí* | No |
| `retoque_vertical_vrtlvertical_flux2_klein_4b.safetensors` | DeverStyle/Flux.2-Klein-Loras | 142MB | No | No |
| `flux-2-klein-4b-zoom-lora.safetensors` | fal/flux-2-klein-4B-zoom-lora | 45MB | No | No |
| `flux2klein_nsfw.safetensors` | ❌ NO FUNCIONA - era para dev | 158MB | Sí | — |

\* Los LoRAs de `DeverStyle` pueden incluir contenido sugerido, pero no pornografía explícita (según su disclaimer).

---

## ⚙️ Configuración Tras Descargar

1. **Ubicar el LoRA**:
   ```text
   D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras\[nombre].safetensors
   ```

2. **Reiniciar la aplicación** (ComfyUI + Gradio)

3. **En la UI Image Editor**:
   - Motor: `flux_klein`
   - LoRA: Selecciona el recién descargado del dropdown
   - Si no aparece, verifica que el archivo esté en la carpeta y sea `.safetensors`

4. **Prompt de ejemplo con LoRA**:
   ```
   [nombre_loro] nude woman, nsfw, explicit content
   ```
   (Revisa el trigger exacto en la página del LoRA en HuggingFace)

---

## 🐛 Solución de Problemas

### Error: `shape mismatch` o `invalid input size`
- **Causa**: LoRA incompatible (para FLUX.1-dev o FLUX.1-schnell)
- **Solución**: Usa solo LoRAs < 350MB y con `klein` en el nombre.

### Error: `LoRA key not loaded`
- **Causa**: El LoRA no tiene claves para el modelo actual
- **Solución**: Verifica que el LoRA sea para `flux_klein` (no `flux_dev`)

### El LoRA no aparece en el dropdown
- **Causa**: Archivo en carpeta incorrecta o extensión wrong
- **Solución**:
  ```powershell
  # Debe estar en:
  D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras\
  
  # Y terminar en .safetensors (no .pt, no .ckpt)
  ```

---

## 💡 Recomendación Final

**Para NSFW rápido y confiable**:
1. Usa el **modelo base** + text encoder uncensored (ya configurado)
2. No uses LoRA (evitas problemas de compatibilidad)
3. Prompts claros: `nude, explicit nsfw, adult content`

**Si quieres un estilo específico** (ej. "solo caricias", "estilo artístico"):
1. Descarga un LoRA pequeño (<300MB) de `DeverStyle/Flux.2-Klein-Loras`
2. Usa el trigger del LoRA en el prompt (ej. `[alba_baptista] nude...`)

---

## 📞 Soporte

Si necesitas un LoRA específico que no está en la lista:
1. Abre un issue en el repositorio del proyecto
2. Proporciona el enlace HuggingFace del LoRA deseado
3. Verificamos compatibilidad antes de descargar
