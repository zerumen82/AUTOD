# ComfyUI PyTorch Setup - RESUELTO

## Problema Original
Error al iniciar ComfyUI: `RuntimeError: config[i] == get()->name() INTERNAL ASSERT FAILED`

## Solución Encontrada

El fix es usar el allocator alternativo de CUDA:
```
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```

## Configuración Actual

### venv_flux (GPU) - USADO POR DEFECTO
- **Ubicación**: `D:\PROJECTS\AUTOAUTO\venv_flux`
- **PyTorch**: 2.4.1+cu124
- **CUDA**: ✅ Funcionando con RTX 3060 Ti
- **Allocator**: cudaMallocAsync (fix aplicado)

### ComfyUI
- Usa venv_flux automáticamente
- Puerto: 8188
- GPU: **FUNCIONANDO** ✅

## Estado Verificado
```
[ComfyUI] Device: cuda:0 NVIDIA GeForce RTX 3060 Ti : cudaMallocAsync
[ComfyUI] Set vram state to: NORMAL_VRAM
[ComfyUI] Using pytorch attention
```

## Cómo Funciona
1. El launcher usa `venv_flux` por defecto
2. Aplica `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` como variable de entorno
3. ComfyUI inicia con GPU sin el bug del allocator
