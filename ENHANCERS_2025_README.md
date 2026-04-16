# 🎭 Face Enhancer Models - 2025 Update

## 📋 Resumen de Cambios

### ✅ Modelos MANTENIDOS (Mejores 2025)

| Modelo | Estado | Calidad (FID↓) | Identidad (LMD↓) | Uso |
|--------|--------|---------------|------------------|-----|
| **CodeFormer** | ✅ **DEFAULT** | 38.13 | **5.41** | Mejor calidad general |
| **RestoreFormer++** | ✅ Disponible | 38.41 | 8.52 | Alta calidad alternativo |

### ❌ Modelos ELIMINADOS (Obsoletos)

| Modelo | Estado | Razón |
|--------|--------|-------|
| **GFPGAN** | ❌ Eliminado | Obsoleto desde 2023 - Peor calidad (FID 42.62) |
| **GPEN** | ❌ Eliminado | Obsoleto desde 2023 - Peor calidad (FID 59.70) |

---

## 🚀 Descarga de Modelos

### Opción 1: Script Automático (Recomendado)

```bash
# Descarga CodeFormer automáticamente
python install_enhancer_models.py

# O usa el script interactivo
python download_enhancer_models.py
```

### Opción 2: Descarga Manual con huggingface-cli (Recomendado)

```bash
# Instalar huggingface-cli si no lo tienes
pip install huggingface_hub

# Descargar CodeFormer
huggingface-cli download sczhou/CodeFormer CodeFormerv0.1.onnx \
  --local-dir ./roop/models/CodeFormer --local-dir-use-symlinks False

# Descargar RestoreFormer++
huggingface-cli download sczhou/CodeFormer restoreformer_plus_plus.onnx \
  --local-dir ./roop/models --local-dir-use-symlinks False
```

### Opción 3: Descarga Manual desde GitHub (Alternativa)

CodeFormer también está disponible en GitHub:

```bash
# CodeFormer desde GitHub (alternativa si HuggingFace falla)
curl -L https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/CodeFormerv0.1.onnx \
  -o ./roop/models/CodeFormer/CodeFormerv0.1.onnx
```

### Opción 4: Descarga Manual desde Navegador

1. **CodeFormer**:
   - Ve a: https://huggingface.co/sczhou/CodeFormer
   - Click en "Files and versions"
   - Descarga `CodeFormerv0.1.onnx`
   - Guarda en: `roop/models/CodeFormer/CodeFormerv0.1.onnx`

2. **RestoreFormer++**:
   - Ve a: https://huggingface.co/sczhou/CodeFormer
   - Descarga `restoreformer_plus_plus.onnx`
   - Guarda en: `roop/models/restoreformer_plus_plus.onnx`

---

## ⚙️ Configuración por Defecto 2025

### UI Principal (faceswap_tab.py)

```python
# Enhancer por defecto
ui_selected_enhancer = "CodeFormer"  # Antes: "GPEN"

# Blend ratio del enhancer
enhancer_blend_factor = 0.3  # 30% enhancer, 70% original

# Modo rápido (sin enhancer)
ui_selected_enhancer = "None"
```

### Configuraciones por Subcategoría

| Subcategoría | Enhancer | Uso |
|--------------|----------|-----|
| **General** | CodeFormer | Uso diario equilibrado |
| **Acciones de Boca** | CodeFormer | Preserva expresiones |
| **Expresiones Faciales** | CodeFormer | Máxima naturalidad |
| **Modo Rápido** | None | Máxima velocidad |
| **Modo Alta Calidad** | RestoreFormer++ | Máxima calidad |

---

## 📊 Comparativa Técnica

### Calidad de Restauración (FID - Menor es Mejor)

```
CodeFormer++      ████████████████████░░  38.13 ⭐
RestoreFormer++   ████████████████████░░  38.41
GFPGAN v1.4       ██████████████████████░ 42.62 ❌
GPEN-BFR-512      ████████████████████████████ 59.70 ❌
```

### Preservación de Identidad (LMD - Menor es Mejor)

```
CodeFormer++      ██████████░░░░░░░░░░░░  5.41  ⭐⭐⭐
GPEN-BFR-512      ██████████████░░░░░░░░  7.26  ⭐⭐
RestoreFormer++   ████████████████░░░░░░  8.52  ⭐⭐
GFPGAN v1.4       ███████████████████░░░  9.50  ⭐
```

---

## 🎯 Recomendaciones de Uso

### ✅ CodeFormer (DEFAULT)
- **Cuándo usar**: Siempre, es el mejor en 2025
- **Ventajas**:
  - Mejor preservación de identidad (LMD 5.41)
  - Excelente calidad visual (FID 38.13)
  - Balance perfecto calidad/rendimiento
- **Blend recomendado**: 0.3 (30% enhancer)

### ✅ RestoreFormer++
- **Cuándo usar**: Modo alta calidad, cuando CodeFormer no es suficiente
- **Ventajas**:
  - Calidad visual similar a CodeFormer
  - Ligeramente más rápido en algunos casos
- **Blend recomendado**: 0.25-0.3

### ❌ GFPGAN / GPEN (ELIMINADOS)
- **Estado**: Obsoletos, eliminados del proyecto
- **Razón**: Peor calidad y preservación de identidad
- **Alternativa**: Usar CodeFormer o RestoreFormer++

---

## 🔧 Configuración Técnica

### ProcessMgr.py

```python
# Enhancers disponibles (2025)
from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus

# Default
selected_enhancer = 'CodeFormer'  # Antes: 'GFPGAN'
```

### globals.py

```python
# Configuración óptima 2025
default_enhancer = 'CodeFormer'
enhancer_blend_factor = 0.3  # 30% enhancer, 70% original
```

---

## 📝 Notas de la Actualización

### Cambios Realizados
1. ✅ Eliminado GFPGAN de UI y código
2. ✅ Eliminado GPEN de UI y código
3. ✅ CodeFormer establecido como default
4. ✅ Actualizados todos los presets
5. ✅ Script de descarga creado

### Archivos Modificados
- `ui/tabs/faceswap_tab.py` - UI actualizada
- `roop/globals.py` - Defaults actualizados
- `roop/ProcessMgr.py` - Procesadores actualizados
- `download_enhancer_models.py` - Nuevo script de descarga

### Beneficios
- 🎯 **Mejor calidad**: CodeFormer tiene FID 38.13 vs 59.70 de GPEN
- 👤 **Mejor identidad**: LMD 5.41 vs 9.50 de GFPGAN
- ⚡ **Más rápido**: Menos modelos obsoletos que cargar
- 🧹 **Más limpio**: Código simplificado sin modelos viejos

---

## 🔗 Referencias

- [CodeFormer Paper](https://arxiv.org/abs/2204.11840)
- [CodeFormer++ Paper (2025)](https://arxiv.org/html/2510.04410v1)
- [RestoreFormer++ Paper](https://arxiv.org/abs/2209.14782)
- [Comparativa FID/LMD](https://github.com/sczhou/CodeFormer)

---

**Última actualización**: Marzo 2025
**Versión**: 2025.1
