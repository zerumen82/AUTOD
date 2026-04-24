# Estado del Sistema Video AI - ACTUALIZADO

## Modelos Disponibles (VERIFICADO)
- [OK] **SVD Turbo**: `StableDiffusionTurbo/` (contiene svd_xt.safetensors)
- [OK] **Wan2.2-Animate-14B**: `Wan2.2-Animate-14B-Q2_K.gguf` (6.4GB)
- [OK] **Zeroscope V2 XL**: `zeroscope_v2_XL/` (formato Diffusers)

## Dropdown
- [OK] `show_label=False` configurado (sin titular)

## Problemas Pendientes
1. **ComfyUI no está ejecutándose** - Necesitas iniciarlo
2. **Nodos faltantes** - GGUFModelLoader y DiffusersLoader no confirmados

## Para que funcione:
1. Inicia ComfyUI desde la UI o con: `python ui/tob/ComfyUI/main.py`
2. Espera a que ComfyUI esté en http://127.0.0.1:8188
3. Verifica los nodos disponibles
4. Instala nodos faltantes via ComfyUI Manager

## Comandos para verificar:
```bash
# Iniciar ComfyUI
cd d:/PROJECTS/AUTOAUTO
python ui/tob/ComfyUI/main.py

# Verificar estado (en otro terminal)
curl http://127.0.0.1:8188/system_stats
```
