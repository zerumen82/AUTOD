# 📋 Checklist de Mejoras: FaceSwap e Image Editor

Edita este archivo añadiendo tus precisiones en cada punto. Cuando termines, avísame en el chat.

---

## 👤 Pestaña FaceSwap (Optimización y UX)

### 1. Refactorización de `faceswap_tab.py`
* **Idea:** Dividir el archivo de +4600 líneas en módulos (ui, logic, processing).
* **[MIS PRECISIONES]:** 

### 2. Propagación Inteligente de Caras
* **Idea:** Auto-selección por embedding facial en todo el lote tras la primera selección.
* **[MIS PRECISIONES]:** 

### 3. Filtros de Galería
* **Idea:** Filtros rápidos por género o tamaño de rostro en fotos grupales.
* **[MIS PRECISIONES]:** 

### 4. Preview de Enhancer "Ojo"
* **Idea:** Botón para ver un "crop" rápido del antes/después del enhancer.
* **[MIS PRECISIONES]:** 

### 5. Detección de Oclusiones
* **Idea:** Evitar que el swap tape pelo o manos que cruzan la cara.
* **[MIS PRECISIONES]:** 

---

## 🎨 Pestaña Image Editor (Funcionalidad y Control)

### 6. Canvas de Máscaras (Inpaint)
* **Idea:** Pintar a mano el área a modificar sobre la imagen.
* **[MIS PRECISIONES]:** 

### 7. Resolutor de Parámetros (Backend)
* **Idea:** Ocultar o bloquear sliders que no funcionan según el motor (HART, Klein, etc).
* **[MIS PRECISIONES]:** 

### 8. Comparador de Cortina (Before/After)
* **Idea:** Slider visual interactivo para comparar original y resultado.
* **[MIS PRECISIONES]:** 

### 9. Prompt-to-Mask (CLIPSeg)
* **Idea:** Generación automática de máscara escribiendo el objeto (ej. "vestido").
* **[MIS PRECISIONES]:** 

### 10. Fijación de Identidad (Personaje)
* **Idea:** Usar la cara de FaceSwap como referencia para que no cambie al editar.
* **[MIS PRECISIONES]:** 

---

## 🤖 Mejoras de IA Generales

### 11. Análisis de Imagen Mejorado
* **Idea:** Detección de personas y sugerencia de prompts automáticos.
* **[MIS PRECISIONES]:** 

### 12. Cola de Trabajos
* **Idea:** Sistema para mandar varios edits y que se procesen en segundo plano.
* **[MIS PRECISIONES]:** 

---
## 💡 Otras ideas o notas generales
* **[ESCRIBE AQUÍ]:** 
